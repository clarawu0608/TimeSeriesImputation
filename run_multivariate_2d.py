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
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.1)  
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--pred_len", type=int, default=0)
parser.add_argument("--lm", type=int, default=10)
parser.add_argument("--missing_rate", type=float, default=0.25)
parser.add_argument("--token_t_size", type=int, default=8)
parser.add_argument("--token_t_overlap", type=int, default=0)
parser.add_argument("--token_d_size", type=int, default=8)
parser.add_argument("--token_d_overlap", type=int, default=0)
parser.add_argument("--loss_r", type=float, default=0.65)
parser.add_argument("--missing_type", type=int, default=0, choices=[0, 1], help="0: Gaussian missing, 1: Prediction missing")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
epoch_num = 350

# Output path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

# Read data
df = pd.read_csv(args.data_path, parse_dates=["date"])
data = df.iloc[:, 1:].values.astype(np.float32)  # multivariate data
timestamps = df["date"].values

# Normalize for each column
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data = (data - data_mean) / data_std

# Mask generation
def generate_mask_matrix_from_paper(length, lm=5, r=0.15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mask = np.ones(length, dtype=bool)
    lu = int((1 - r) / r * lm)
    i = 0
    while i < length:
        masked_len = np.random.geometric(1 / lm)
        end = min(i + masked_len, length)
        mask[i:end] = False
        i = end
        unmasked_len = np.random.geometric(1 / lu)
        i += unmasked_len
    return mask

# prediction mask generation
def generate_prediction_mask(sequence_len, seq_len, pred_len):
    """
    Generate a mask where:
    - The sequence is divided into chunks of size `seq_len`.
    - For each full chunk, the last `pred_len` elements are masked.
    - For the last partial chunk (if any), mask `int(round(chunk_len * (pred_len / seq_len)))` elements.
    
    Args:
        sequence_len: Total length of the mask.
        seq_len: Length of each full chunk.
        pred_len: Number of masked elements per full chunk.
    
    Returns:
        Boolean mask (True = unmasked, False = masked).
    """
    mask = np.ones(sequence_len, dtype=bool)
    
    # Process full chunks
    for start in range(0, sequence_len, seq_len):
        end = start + seq_len
        if end > sequence_len:
            break  # handle the last partial chunk separately
        mask[end - pred_len : end] = False
    
    # Process the last partial chunk (if any)
    remaining = sequence_len % seq_len
    if remaining > 0:
        last_chunk_start = sequence_len - remaining
        last_chunk_len = remaining
        prop_masked = int(round(last_chunk_len * (pred_len / seq_len)))
        prop_masked = max(0, min(prop_masked, last_chunk_len))  # ensure valid range
        if prop_masked > 0:
            mask[last_chunk_start + last_chunk_len - prop_masked : sequence_len] = False
    
    return mask

def generate_prediction_mask_fixed(seq_len, pred_len):
    """
    Generate a mask for a fixed sequence length where the last `pred_len` elements are masked.
    
    Args:
        seq_len (int): Total length of the sequence.
        pred_len (int): Number of elements to mask at the end.
    
    Returns:
        np.ndarray: Boolean mask (True = unmasked, False = masked).
    """
    mask = np.ones(seq_len, dtype=bool)
    if pred_len > 0:
        mask[-pred_len:] = False
    return mask

T, D = data.shape
mask_matrix = np.ones((T, D), dtype=bool)
if args.missing_type == 0:
    total = T * D
    missing_indices = np.random.choice(total, int(total * args.missing_rate), replace=False)
    mask_matrix.flat[missing_indices] = False
elif args.missing_type == 1:
    for d in range(D):
        if args.missing_type == 0:
            mask_matrix[:, d] = generate_mask_matrix_from_paper(T, lm=args.lm, r=args.missing_rate, seed=42 + d)
        elif args.missing_type == 1:
            mask_matrix[:, d] = generate_prediction_mask(T, seq_len=args.seq_len, pred_len=args.lm)

masked_data = data.copy()
masked_data[~mask_matrix] = 0.0

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

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, masked_data, mask, seq_len):
        self.data = data
        self.masked_data = masked_data
        self.mask = mask
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx):
        if args.missing_type == 0:
            mask = self.mask[idx:idx+self.seq_len]
        elif args.missing_type == 1:
            mask = np.zeros((args.seq_len, self.data.shape[1]), dtype=bool)
            for d in range(self.data.shape[1]):
                mask[:, d] = generate_prediction_mask_fixed(seq_len=args.seq_len, pred_len=args.pred_len)
        return (
            self.data[idx:idx+self.seq_len],
            self.masked_data[idx:idx+self.seq_len],
            mask
        )

full_dataset = TimeSeriesDataset(data, masked_data, mask_matrix, args.seq_len)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=True)

# Patch2D Embedding
class Patch2DEmbedding(nn.Module):
    def __init__(self, token_t_size, token_t_overlap, token_d_size, token_d_overlap, d_model):
        super().__init__()
        self.token_t_size = token_t_size
        self.token_t_overlap = token_t_overlap
        self.token_d_size = token_d_size
        self.token_d_overlap = token_d_overlap
        # print(f"Patch2DEmbedding: token_t_size={token_t_size}, token_s_size={token_d_size}, d_model={d_model}")
        self.linear = nn.Linear(token_t_size * token_d_size, d_model)

    def forward(self, x):  # (B, T, D)
        B, T, D = x.shape
        patches = []
        for t_start in range(0, T - self.token_t_size + 1, self.token_t_size - self.token_t_overlap):
            for d_start in range(0, D - self.token_d_size + 1, self.token_d_size - self.token_d_overlap):
                patch = x[:, t_start:t_start+self.token_t_size, d_start:d_start+self.token_d_size]
                patches.append(patch.reshape(B, -1))
        patches = torch.stack(patches, dim=1)  # (B, N_patches, token_t_size*token_d_size)
        return self.linear(patches)  # (B, N_patches, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=args.nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.attn_weights = None

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        self.attn_weights = attn_weights.detach()
        src = self.norm1(src + self.dropout1(attn_output))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))
    
def reconstruct_from_patches(patches, B, T, D, token_t_size, token_t_overlap, token_d_size, token_d_overlap):
    recon = torch.zeros((B, T, D), device=patches.device)
    count = torch.zeros((B, T, D), device=patches.device)

    idx = 0
    for t_start in range(0, T - token_t_size + 1, token_t_size - token_t_overlap):
        for d_start in range(0, D - token_d_size + 1, token_d_size - token_d_overlap):
            patch = patches[:, idx].reshape(B, token_t_size, token_d_size)
            recon[:, t_start:t_start+token_t_size, d_start:d_start+token_d_size] += patch
            count[:, t_start:t_start+token_t_size, d_start:d_start+token_d_size] += 1
            idx += 1

    return recon / count.clamp(min=1)

class MultiScaleTwoDTransformerEncoder(nn.Module):
    def __init__(self, args=args):
        super().__init__()
        self.args = args

        self.token_t_sizes = [int(args.seq_len / 16), int(args.seq_len / 8), int(args.seq_len / 4), int(args.seq_len / 2)]
        self.token_d_size = args.token_d_size
        self.token_t_overlap = args.token_t_overlap
        self.token_d_overlap = args.token_d_overlap
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        self.mask_token = nn.Parameter(torch.randn(1))

        self.branches = nn.ModuleList()

        for token_t_size in self.token_t_sizes:
            patch_size = token_t_size * self.token_d_size
            d_model = token_t_size * self.nhead  # consistent with your strategy

            branch = nn.ModuleDict({
                'embed': Patch2DEmbedding(
                    token_t_size=token_t_size,
                    token_t_overlap=self.token_t_overlap,
                    token_d_size=self.token_d_size,
                    token_d_overlap=self.token_d_overlap,
                    d_model=d_model
                ),
                'pos_encoder': PositionalEncoding(d_model),
                'layers': nn.ModuleList([
                    CustomTransformerEncoderLayer(d_model=d_model, nhead=self.nhead)
                    for _ in range(self.num_layers)
                ]),
                'project': nn.Linear(d_model, patch_size)
            })

            self.branches.append(branch)

        self.alpha = nn.Parameter(torch.ones(len(self.token_t_sizes)))  # raw combination weights

    def forward(self, x, mask):
        B, T, D = x.shape
        x = torch.where(mask, x, self.mask_token.expand_as(x))  # masking

        branch_outputs = []
        weights = torch.softmax(self.alpha, dim=0)

        for i, token_t_size in enumerate(self.token_t_sizes):
            branch = self.branches[i]

            # Embedding and positional encoding
            x_embed = branch['embed'](x)              # (B, N_patches, d_model)
            x_embed = branch['pos_encoder'](x_embed)  # (B, N_patches, d_model)

            for layer in branch['layers']:
                x_embed = layer(x_embed)              # (B, N_patches, d_model)

            x_proj = branch['project'](x_embed)       # (B, N_patches, patch_size)

            # Reconstruct to (B, T, D)
            recon = reconstruct_from_patches(
                x_proj,
                B=B, T=T, D=D,
                token_t_size=token_t_size,
                token_t_overlap=self.token_t_overlap,
                token_d_size=self.token_d_size,
                token_d_overlap=self.token_d_overlap
            )
            branch_outputs.append(recon)

        # Weighted combination of outputs
        out = sum(w * o for w, o in zip(weights, branch_outputs))  # (B, T, D)
        return out

def denormalize(x, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    if x.ndim == 2:
        if x.shape[1] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[1] == mean.shape[0], got {x.shape[1]} and {mean.shape[0]}")
        return x * std[np.newaxis, :] + mean[np.newaxis, :]
    elif x.ndim == 3:
        if x.shape[2] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[2] == mean.shape[0], got {x.shape[2]} and {mean.shape[0]}")
        return x * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    elif x.ndim == 1:
        if x.shape[0] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[0] == mean.shape[0], got {x.shape[0]} and {mean.shape[0]}")
        return x * std + mean
    else:
        raise ValueError(f"Unsupported x shape: {x.shape}")


# 2D Model initialization
model = MultiScaleTwoDTransformerEncoder().to(device)
criterion = MaskedWeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

twoD_output_path = output_path + "2D/"
os.makedirs(twoD_output_path, exist_ok=True)

# Training and Validation
best_val_loss = float('inf')
early_stop_counter = 0
train_losses, val_losses = [], []

for epoch in range(epoch_num):
    model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        optimizer.zero_grad()
        recon = model(masked, mask)
        loss = criterion(recon, gt, mask, args.loss_r)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            recon = model(masked, mask)
            loss = criterion(recon, gt, mask, args.loss_r)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    print(f"[2D] Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    scheduler.step()

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        early_stop_counter = 0
        torch.save(model.state_dict(), f"{twoD_output_path}/best_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("[2D] Early stopping")
            break

# Plot loss curve
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{twoD_output_path}/loss_curve.png")



# Load best model for evaluation
model.load_state_dict(torch.load(f"{twoD_output_path}/best_model.pth"))
model.eval()

# Evaluation
mse_total, count = 0.0, 0
for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
    with torch.no_grad():
        out = model(masked, mask)

    gt_np = gt.cpu().numpy().squeeze()
    if gt_np.ndim == 1:
        gt_np = gt_np[np.newaxis, :]
    out_np = out.cpu().numpy().squeeze()
    if out_np.ndim == 1:
        out_np = out_np[np.newaxis, :]
    mask_np = mask.cpu().numpy().squeeze()
    if mask_np.ndim == 1:
        mask_np = mask_np[np.newaxis, :]


    gt_denorm = denormalize(gt_np, data_mean, data_std)
    out_denorm = denormalize(out_np, data_mean, data_std)

    mse = ((gt_np[~mask_np] - out_np[~mask_np])**2).sum()
    mse_total += mse
    count += (~mask_np).sum()

    if i < 2:
        os.makedirs(f"{twoD_output_path}/test_case_{i}/", exist_ok=True)
        for j in range (0, D):
            plt.figure()
            plt.plot(gt_denorm[0, :, j], label="GT")
            plt.plot(out_denorm[0, :, j], label="Output")
            plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
            plt.legend()
            plt.title(f"Baseline Test Case {i} column {j}")
            plt.savefig(f"{twoD_output_path}/test_case_{i}/column_{j}.png")

print("[2D] Masked MSE on test set:", mse_total / count)
with open("test_results.txt", "a") as log_file:
    print("[2D] Writing to file...")
    log_file.write("[2D] Masked MSE on test set:" + str(mse_total / count) + "\n" + "\n" + "\n")
