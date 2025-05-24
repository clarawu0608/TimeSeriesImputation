import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str)
parser.add_argument("--output_path", required=True, type=str)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--nhead", type=int, default=1)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--token_size", type=int, default=8)
parser.add_argument("--token_overlap", type=int, default=0)
parser.add_argument("--missing_type", type=int, default=1)
parser.add_argument("--lm", type=int, default=15)
parser.add_argument("--missing_rate", type=float, default=0.3)
parser.add_argument("--loss_r", type=float, default=0.8)
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Output path
output_path = args.output_path 
os.makedirs(output_path, exist_ok=True)

# Read data
df = pd.read_csv(args.data_path, parse_dates=["date"])
data = df.iloc[:, 1].values.astype(np.float32)

# Normalize data
data_mean = np.mean(data)
data_std = np.std(data)
data = (data - data_mean) / data_std

# Mask generation
def generate_mask_matrix(length, missing_rate):
    mask = np.ones(length, dtype=bool)
    missing_indices = np.random.choice(length, int(length * missing_rate), replace=False)
    mask[missing_indices] = False
    return mask

def generate_mask_matrix_from_paper(length, lm=5, r=0.15, seed=None):
    """
    Generate a mask using the geometric masking strategy described in the paper.

    Parameters:
    - length (int): total length of the sequence
    - lm (float): mean masked segment length
    - r (float): masking rate, used to compute unmasked length
    - seed (int): random seed for reproducibility

    Returns:
    - mask (np.ndarray): boolean mask (True = observed, False = masked)
    """
    if seed is not None:
        np.random.seed(seed)

    if r == 0:
        return np.ones(length, dtype=bool)

    mask = np.ones(length, dtype=bool)

    lu = int((1 - r) / r * lm)  # mean unmasked length
    i = 0

    while i < length:
        # Masked segment
        masked_len = np.random.geometric(1 / lm)
        # print(f"Masked length: {masked_len}")
        end = min(i + masked_len, length)
        mask[i:end] = False
        i = end

        # Unmasked segment
        unmasked_len = np.random.geometric(1 / lu)
        # print(f"Unmasked length: {unmasked_len}")
        i += unmasked_len

    return mask

# criterion
class MaskedWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # element-wise loss

    def forward(self, out, gt, mask, r):
        mse_loss = self.mse(out, gt)  # shape: [batch, seq_len, ...]
        
        # Convert mask to float for computation
        mask = mask.float()
        
        # Masked positions (0 in mask)
        masked_loss = mse_loss * (1 - mask) * r * 2
        
        # Unmasked positions (1 in mask)
        unmasked_loss = mse_loss * mask * (1 - r) * 2
        
        total_loss = masked_loss + unmasked_loss
        
        # Final scalar loss (mean over all elements)
        return total_loss.mean()

if args.missing_type == 0:
    mask_matrix = generate_mask_matrix(len(data), args.missing_rate)
elif args.missing_type == 1:
    mask_matrix = generate_mask_matrix_from_paper(len(data), lm=args.lm, r=args.missing_rate, seed=42)
masked_data = data.copy()
masked_data[~mask_matrix] = 0.0

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, masked_data, mask, seq_len):
        self.data = data
        self.masked_data = masked_data
        self.mask = mask
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.data[idx:idx+self.seq_len],
            self.masked_data[idx:idx+self.seq_len],
            self.mask[idx:idx+self.seq_len]
        )

full_dataset = TimeSeriesDataset(data, masked_data, mask_matrix, args.seq_len)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
train_indices = train_set.indices
val_indices = val_set.indices
test_indices = test_set.indices

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

# Transformer model with attention map capture
class PatchEmbedding(nn.Module):
    def __init__(self, token_size, d_model):
        super().__init__()
        self.linear = nn.Linear(token_size, d_model)

    def forward(self, x):
        return self.linear(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"PositionalEncoding max_len={self.pe.size(1)} is too small for input sequence length {x.size(1)}.")
        return x + self.pe[:, :x.size(1)].to(x.device)
    

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(args.dropout_rate)
        self.dropout2 = nn.Dropout(args.dropout_rate)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                    key_padding_mask=src_key_padding_mask,
                                                    need_weights=True)
        self.attn_weights = attn_weights.detach()
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
def reconstruct_from_vertical_patches(temporal_tokens, B, T, token_t_size, token_t_overlap):
    # temporal_tokens: (B, N_t * D, token_t_size)
    
    recon = torch.zeros((B, T), device=device)
    count = torch.zeros((B, T), device=device)

    t_starts = list(range(0, T - token_t_size + 1, token_t_size - token_t_overlap))

    for i, t_start in enumerate(t_starts):
        patch = temporal_tokens[:,i, :].reshape(B, token_t_size, 1).squeeze(-1)  # (B, token_t_size)
        recon[:, t_start:t_start+token_t_size] += patch
        count[:, t_start:t_start+token_t_size] += 1

    return recon / count.clamp(min=1e-6) # (B, T, D)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=args.token_size, nhead=args.nhead, num_layers=args.num_layers):
        super().__init__()
        self.embed = PatchEmbedding(args.token_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.project = nn.Linear(args.token_size, args.token_size)
        self.attn_weights = None
        self.mask_token = nn.Parameter(torch.randn(1))
        num_patches = ((args.seq_len - args.token_size) // (args.token_size - args.token_overlap)) + 1
        self.reconstructor = nn.Linear(d_model * num_patches, args.seq_len)

    def forward(self, x, mask=None):
        x = torch.where(mask, x, self.mask_token.expand_as(x))
        B, L = x.shape
        patches = x.unfold(1, args.token_size, args.token_size - args.token_overlap) # (B, num_patches, token_size)
        patches = self.embed(patches) # (B, num_patches, d_model)
        x = self.pos_encoder(patches)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)
            self.attn_weights = layer.attn_weights # (B, num_patches, d_model)

        x = x.reshape(B, -1) # (B, num_patches * d_model)
        out = self.reconstructor(x) # (B, seq_len)
        # out = reconstruct_from_vertical_patches(x, B, args.seq_len, args.token_size, args.token_overlap) # (B, seq_len)
        
        return out

model = TransformerEncoder().to(device)
criterion = MaskedWeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 250
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

best_val_loss = float('inf')
early_stop_counter = 0
train_losses, val_losses = [], []

# Training
for epoch in range(epoch_num):
    model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(masked, mask)
        loss = criterion(out, gt, mask, args.loss_r)
        loss.backward()
        optimizer.step(), mask
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = model(masked, mask)
            loss = criterion(out, gt, mask, args.loss_r)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    scheduler.step()

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        early_stop_counter = 0
        torch.save(model.state_dict(), args.output_path + "best_model.pth")
        last_attn_weights = model.attn_weights
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping")
            break


# Save loss curves
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{output_path}loss_curve.png")

# Plot one attention map
for i in range(len(model.layers)):
    if hasattr(model.layers[i], 'attn_weights') and model.layers[i].attn_weights is not None:
        attn_map = model.layers[-1].attn_weights[0].cpu().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title("Attention Map Layer {}".format(i))
        plt.savefig(f"{output_path}attention_map_layer{i}.png")

# Test and baseline comparison

class RNNImputer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        out, _ = self.lstm(x)
        out = self.output(out)
        return out.squeeze(-1)  # [B, L, 1] -> [B, L]

# -------------------------
# RNN Imputer Training
# -------------------------
print("\nTraining LSTM Imputer...")
rnn_model = RNNImputer().to(device)
rnn_criterion = nn.MSELoss()
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
rnn_epoch_num = 100
rnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rnn_optimizer, T_max=rnn_epoch_num)

best_rnn_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(rnn_epoch_num):
    rnn_model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        rnn_optimizer.zero_grad()
        out = rnn_model(masked)
        loss = rnn_criterion(out, gt)
        loss.backward()
        rnn_optimizer.step()
        train_loss += loss.item()

    rnn_model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = rnn_model(masked)
            loss = rnn_criterion(out, gt)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"[LSTM Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    rnn_scheduler.step()

    if avg_val_loss < best_rnn_val_loss:
        best_rnn_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(rnn_model.state_dict(), args.output_path + "best_rnn_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping for LSTM")
            break

# -------------------------
# RNN Imputer Evaluation
# -------------------------

# For denormalization
def denormalize(x, data_std, data_mean):
    return x * data_std + data_mean

# For reconstructing overlapping predictions (assuming summing and counting)
def reconstruct(pred_patches, count_map):
    return np.divide(pred_patches, count_map, out=np.zeros_like(pred_patches), where=count_map != 0)

print("\nEvaluating LSTM Imputer on test set...")
rnn_model.load_state_dict(torch.load(args.output_path + "best_rnn_model.pth"))
rnn_model.eval()
mse_total_rnn, count_rnn = 0, 0

for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
    with torch.no_grad():
        out = rnn_model(masked)

    gt_np = gt[0].cpu().numpy()
    out_np = out[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    gt_denorm = denormalize(gt_np, data_std, data_mean)
    out_denorm = denormalize(out_np, data_std, data_mean)
    
    if args.missing_rate != 0:
        mse_total_rnn += ((out_np[~mask_np] - gt_np[~mask_np]) ** 2).sum()
        count_rnn += (~mask_np).sum()
    else:
        mse_total_rnn += ((out_np - gt_np) ** 2).sum()
        count_rnn += (mask_np).sum()

    if i < 2:
        plt.figure()
        plt.plot(gt_denorm, label="GT")
        plt.plot(out_denorm, label="LSTM Output")
        plt.plot(mask_np * max(gt_denorm), label="Mask")
        plt.legend()
        plt.title(f"LSTM Test Case {i}")
        plt.savefig(f"{output_path}lstm_test_case_{i}.png")

lstm_mse = mse_total_rnn / count_rnn
print("LSTM Masked MSE on test set:", lstm_mse)

with open("test_results.txt", "a") as log_file:
    log_file.write(f"LSTM Masked MSE on test set: {lstm_mse:.6f}\n")





model.load_state_dict(torch.load(args.output_path + "best_model.pth"))
model.eval()
mse_total, count = 0, 0

for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
    with torch.no_grad():
        out = model(masked, mask)

    # Convert tensors to numpy
    gt_np = gt[0].cpu().numpy()
    out_np = out[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    # Denormalize
    gt_denorm = denormalize(gt_np, data_std, data_mean)
    out_denorm = denormalize(out_np, data_std, data_mean)

    # Compute masked MSE
    if args.missing_rate != 0:
        mse_total += ((out_np[~mask_np] - gt_np[~mask_np]) ** 2).sum()
        count += (~mask_np).sum()
    else:
        mse_total += ((out_np - gt_np) ** 2).sum()
        count += (mask_np).sum()
 
    # Plotting
    if i < 2:
        plt.figure()
        plt.plot(gt_denorm, label="GT")
        plt.plot(out_denorm, label="Output")
        plt.plot(mask_np * max(gt_denorm), label="Mask")
        plt.legend()
        plt.title(f"Test Case {i}")
        plt.savefig(f"{output_path}test_case_{i}.png")

print("Masked MSE on test set:", mse_total / count)
print("Working directory:", os.getcwd())
with open("test_results.txt", "a") as log_file:
    print("Writing to file...")
    log_file.write(output_path + "\n")
    log_file.write("Masked MSE on test set:" + str(mse_total / count) + "\n")
    
# Baseline interpolation across all test samples
x = np.arange(args.seq_len)

for kind in ["linear", "nearest"]:
    total_mse, valid_count = 0.0, 0
    for i in range(len(test_set)):
        gt, masked, mask = test_set[i]
        if (~mask).sum() == 0:
            continue
        masked_interp = masked.copy()
        masked_interp[~mask] = np.nan
        try:
            f = interp1d(x[mask], gt[mask], kind=kind, fill_value='extrapolate')
            interp_values = masked_interp.copy()
            interp_values[~mask] = f(x[~mask])
            mse_interp = np.mean((gt[~mask] - interp_values[~mask])**2)
            total_mse += mse_interp
            valid_count += 1
        except Exception as e:
            print(f"{kind.capitalize()} interpolation failed on sample {i}: {e}")
            continue

    if valid_count > 0:
        avg_mse = total_mse / valid_count
    else:
        avg_mse = float('nan')

    print(f"{kind.capitalize()} interpolation average MSE over test set: {avg_mse:.6f}")
    with open("test_results.txt", "a") as log_file:
        log_file.write(f"{kind.capitalize()} interpolation average MSE: {avg_mse:.6f}\n")

with open("test_results.txt", "a") as log_file:
    log_file.write("\n")