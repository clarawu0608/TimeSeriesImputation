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
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--token_size", type=int, default=8)
parser.add_argument("--token_overlap", type=int, default=0)
parser.add_argument("--epoch_num", type=int, default=100)
parser.add_argument("--missing_type", type=int, default=1)
parser.add_argument("--lm", type=int, default=10)
parser.add_argument("--missing_rate", type=float, default=0.25)
args = parser.parse_args()

# Output path
output_path = "outputs/period_{}_{}_{}_{}_{}_{}/".format(
    args.seq_len, args.token_size, args.token_overlap, args.missing_type, args.lm, args.missing_rate)
os.makedirs(output_path, exist_ok=True)

# Read data
df = pd.read_csv("dataset/periodic_signal.csv", parse_dates=["date"])
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
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=1)

# Transformer model with attention map capture
class PatchEmbedding(nn.Module):
    def __init__(self, token_size):
        super().__init__()
        self.token_size = token_size
        self.linear = nn.Linear(token_size, token_size)

    def forward(self, x):
        B, L = x.shape
        patches = x.unfold(1, self.token_size, self.token_size - args.token_overlap)
        patches = self.linear(patches)
        return patches
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
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

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=8):
        super().__init__()
        self.embed = PatchEmbedding(args.token_size)
        self.pos_encoder = PositionalEncoding(d_model=args.token_size)
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(args.token_size, nhead) for _ in range(num_layers)])
        self.project = nn.Linear(args.token_size, args.token_size)
        self.attn_weights = None
        self.mask_token = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = torch.where(mask, x, self.mask_token.expand_as(x))
        B, L = x.shape
        patches = x.unfold(1, args.token_size, args.token_size - args.token_overlap)
        patches = self.embed.linear(patches)
        x = self.pos_encoder(patches)

        for layer in self.layers:
            x = layer(x)
            self.attn_weights = layer.attn_weights

        out = self.project(x).reshape(B, -1)
        if out.shape[1] > L:
            out = out[:, :L]
        elif out.shape[1] < L:
            pad = torch.zeros((B, L - out.shape[1]), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out

model = TransformerEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)

best_val_loss = float('inf')
early_stop_counter = 0
train_losses, val_losses = [], []

# Training
for epoch in range(args.epoch_num):
    model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        optimizer.zero_grad()
        out = model(masked)
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            out = model(masked)
            loss = criterion(out, gt)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    scheduler.step()

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
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
if last_attn_weights is not None:
    plt.figure(figsize=(6, 5))
    plt.imshow(last_attn_weights[0].cpu(), cmap='viridis')
    plt.colorbar()
    plt.title("Attention Map")
    plt.savefig(f"{output_path}attention_map.png")

# Test and baseline comparison
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
mse_total, count = 0, 0

for i, (gt, masked, mask) in enumerate(test_loader):
    with torch.no_grad():
        out = model(masked)
    mse_total += ((out - gt)[~mask]).pow(2).sum().item()
    count += (~mask).sum().item()
    if i < 2:
        plt.figure()
        plt.plot(gt[0], label="GT")
        plt.plot(out[0], label="Output")
        plt.plot(mask[0] * max(gt[0]), label="Mask")
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

for kind in ["linear", "cubic"]:
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