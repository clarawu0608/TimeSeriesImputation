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
parser.add_argument("--missing_type", type=int, default=1)
parser.add_argument("--lm", type=int, default=10)
parser.add_argument("--missing_rate", type=float, default=0.25)
parser.add_argument("--token_t_size", type=int, default=8)
parser.add_argument("--token_t_overlap", type=int, default=0)
parser.add_argument("--token_d_size", type=int, default=8)
parser.add_argument("--token_d_overlap", type=int, default=0)
parser.add_argument("--loss_r", type=float, default=0.65)
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
epoch_num = 250

# Output path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

# Read data
df = pd.read_csv(args.data_path, parse_dates=["date"])
df = df.sort_values("date")
data = df.iloc[:, 1:].values.astype(np.float32)  # multivariate data
timestamps = df["date"].values
dt_seconds = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 's')
sampling_rate = 1 / dt_seconds

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

T, D = data.shape
mask_matrix = np.ones((T, D), dtype=bool)
if args.missing_type == 0:
    total = T * D
    missing_indices = np.random.choice(total, int(total * args.missing_rate), replace=False)
    mask_matrix.flat[missing_indices] = False
elif args.missing_type == 1:
    for d in range(D):
        mask_matrix[:, d] = generate_mask_matrix_from_paper(T, lm=args.lm, r=args.missing_rate, seed=42 + d)

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

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=args.nhead, num_layers=args.num_layers):
        super().__init__()
        # Patch2DEmbedding input: (B, T, D)
        # Output: (B, N_patches, d_model)
        self.embed = Patch2DEmbedding(
            args.token_t_size, args.token_t_overlap,
            args.token_d_size, args.token_d_overlap,
            d_model
        )

        # PositionalEncoding input: (B, N_patches, d_model)
        # Output: (B, N_patches, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer layers process (B, N_patches, d_model)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])

        # Final projection maps (B, N_patches, d_model) → (B, N_patches, patch_size)
        # where patch_size = token_t_size * token_d_size
        self.project = nn.Linear(d_model, args.token_t_size * args.token_d_size)

    def forward(self, x, mask):
        # Input: x, mask → (B, T, D)
        x = torch.where(mask, x, 0.0)  # (B, T, D)

        x_embed = self.embed(x)  # → (B, N_patches, d_model)
        x_embed = self.pos_encoder(x_embed)  # → (B, N_patches, d_model)

        for layer in self.layers:
            x_embed = layer(x_embed)  # remains (B, N_patches, d_model)

        out = self.project(x_embed)  # → (B, N_patches, patch_size)
        return out
    

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

# Baseline model
# This model only considers temporal patches, ignoring the spatial dimension

class TimeOnlyEmbedding(nn.Module):
    def __init__(self, token_t_size, token_t_overlap, d_model):
        super().__init__()
        self.token_t_size = token_t_size
        self.token_t_overlap = token_t_overlap
        self.d_model = d_model
        self.linear = nn.Linear(token_t_size, d_model)  # Flatten patch of shape (token_t_size, D)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        temporal_patches = []
        for d_idx in range(0, D):
            for t_start in range(0, T - self.token_t_size + 1, self.token_t_size - self.token_t_overlap):
                patch = x[:, t_start:t_start+self.token_t_size, d_idx]  # (B, token_t_size)
                temporal_patches.append(self.linear(patch))  # (B, d_model)
        temporal_tokens = torch.stack(temporal_patches, dim=1)  # (B, N_t * D, d_model)
        return temporal_tokens  # shape: (B, N_t * D, d_model)

class BaselineTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=args.nhead, num_layers=args.num_layers, token_t_size=args.token_t_size, token_t_overlap=args.token_t_overlap):
        super().__init__()
        self.embed = TimeOnlyEmbedding(
            token_t_size = token_t_size,
            token_t_overlap = token_t_overlap,
            d_model = d_model
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.project = nn.Linear(d_model, token_t_size)
        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, x, mask):
        x = torch.where(mask, x, 0.0)
        x_embed = self.embed(x) # → (B, N_t * D, d_model)
        x_embed = self.pos_encoder(x_embed)
        x_embed = self.dropout(x_embed)
        for layer in self.layers:
            x_embed = layer(x_embed)
        out = self.project(x_embed) # → (B, N_t * D, token_t_size)
        return out


def reconstruct_from_vertical_patches(temporal_tokens, B, T, D, token_t_size, token_t_overlap):
    # temporal_tokens: (B, N_t * D, token_t_size)
    
    recon = torch.zeros((B, T, D), device=device)
    count = torch.zeros((B, T, D), device=device)

    t_starts = list(range(0, T - token_t_size + 1, token_t_size - token_t_overlap))
    n_t = len(t_starts)
    temporal_len = n_t * D

    for d_idx in range(0, D):
        for i, t_start in enumerate(t_starts):
            patch = temporal_tokens[:, d_idx * n_t + i, :].reshape(B, token_t_size, 1).squeeze(-1)  # (B, token_t_size)
            recon[:, t_start:t_start+token_t_size, d_idx] += patch
            count[:, t_start:t_start+token_t_size, d_idx] += 1

    return recon / count.clamp(min=1e-6) # (B, T, D)

# 1D Mixture Model

class Patch1DEmbedding(nn.Module):
    def __init__(self, token_t_size, token_t_overlap, d_model):
        super().__init__()
        self.token_t_size = token_t_size
        self.token_t_overlap = token_t_overlap
        self.linear_t = nn.Linear(token_t_size, d_model)
        self.linear_d = nn.Linear(D, d_model)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        temporal_patches = []
        for d_idx in range(0, D):
            for t_start in range(0, T - self.token_t_size + 1, self.token_t_size - self.token_t_overlap):
                patch = x[:, t_start:t_start+self.token_t_size, d_idx]  # (B, token_t_size)
                temporal_patches.append(self.linear_t(patch))  # (B, d_model)
        temporal_tokens = torch.stack(temporal_patches, dim=1)  # (B, N_t * D, d_model)

        feature_patches = []
        for t_idx in range(0, T):
            patch = x[:, t_idx, :]  # (B, D)
            feature_patches.append(self.linear_d(patch))  # (B, d_model)
        feature_tokens = torch.stack(feature_patches, dim=1)  # (B, T, d_model)

        return torch.cat([temporal_tokens, feature_tokens], dim=1)  # (B, N_t * D + T, d_model)

class MixedTokenTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=args.nhead, num_layers=args.num_layers,
                 token_t_size = args.token_t_size, token_t_overlap = args.token_t_overlap,
                 T = args.seq_len, D = 100):
        super().__init__()
        self.embedding = Patch1DEmbedding(token_t_size, token_t_overlap, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(*[CustomTransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.temporal_len = ((T - token_t_size) // (token_t_size - token_t_overlap) + 1) * D 
        self.feature_len = T
        self.project_t = nn.Linear(d_model, token_t_size)
        self.project_d = nn.Linear(d_model, D)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x, mask):
        x = torch.where(mask, x, 0.0)
        tokens = self.embedding(x)  # (B, N_t * D + T, d_model)
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)
        for layer in self.encoder:
            tokens = layer(tokens)
        temporal_tokens = tokens[:, :self.temporal_len, :]  # (B, N_t * D, d_model)
        feature_tokens = tokens[:, self.temporal_len:, :]   # (B, T, d_model)
        proj_temporal = self.project_t(temporal_tokens)  # (B, N_t * D, token_t_size)
        proj_feature = self.project_d(feature_tokens)    # (B, T, D)
        return proj_temporal, proj_feature  # (B, N_tokens, variable_len)
    

def reconstruct_from_mixed_tokens(device, temporal_tokens, feature_tokens, B, T, D, token_t_size, token_t_overlap):
    # temporal_tokens: (B, N_t * D, token_t_size)
    # feature_tokens: (B, T, D)
    
    recon = torch.zeros((B, T, D), device=device)
    count = torch.zeros((B, T, D), device=device)

    t_starts = list(range(0, T - token_t_size + 1, token_t_size - token_t_overlap))
    n_t = len(t_starts)
    temporal_len = n_t * D

    for d_idx in range(0, D):
        for i, t_start in enumerate(t_starts):
            patch = temporal_tokens[:, d_idx * n_t + i, :].reshape(B, token_t_size, 1).squeeze(-1)  # (B, token_t_size)
            recon[:, t_start:t_start+token_t_size, d_idx] += patch
            count[:, t_start:t_start+token_t_size, d_idx] += 1

    recon = recon + feature_tokens  # Add feature tokens
    count = count + torch.ones((B, T, D), device=device)  # Add count for feature tokens

    return recon / count.clamp(min=1e-6) # (B, T, D)

# Denormalize function
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
    


# Independently imputation
class IndependentPatchEmbedding(nn.Module):
    def __init__(self, token_size, d_model):
        super().__init__()
        self.linear = nn.Linear(token_size, d_model)

    def forward(self, x):
        return self.linear(x)


class IndependentTransformerEncoder(nn.Module):
    def __init__(self, args=args):
        super().__init__()
        self.D = args.token_d_size
        self.d_model = args.token_t_size
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.token_size = args.token_t_size
        self.token_overlap = args.token_t_overlap
        self.seq_len = args.seq_len
        
        # Modules per variate
        self.embed_layers = nn.ModuleList([
            IndependentPatchEmbedding(self.token_size, self.d_model) for _ in range(self.D)
        ])
        self.pos_encoders = nn.ModuleList([
            PositionalEncoding(d_model=self.d_model) for _ in range(self.D)
        ])
        self.transformer_layers = nn.ModuleList([
            nn.ModuleList([
                CustomTransformerEncoderLayer(self.d_model, self.nhead)
                for _ in range(self.num_layers)
            ]) for _ in range(self.D)
        ])
        self.reconstructors = nn.ModuleList([
            nn.Linear(self.d_model * self.num_patches(), self.seq_len) for _ in range(self.D)
        ])
        
        self.mask_token = nn.Parameter(torch.randn(1))
        self.attn_weights = [None for _ in range(self.D)]

    def num_patches(self):
        return ((self.seq_len - self.token_size) // (self.token_size - self.token_overlap)) + 1

    def forward(self, x, mask=None):
        # x: (B, D, L), mask: (B, D, L)
        B, D, L = x.shape
        outputs = []

        for d in range(self.D):
            x_d = x[:, d, :]                   # (B, L)
            mask_d = mask[:, d, :] if mask is not None else torch.ones_like(x_d, dtype=torch.bool)
            x_d = torch.where(mask_d, x_d, self.mask_token.expand_as(x_d))

            # Unfold into patches
            patches = x_d.unfold(1, self.token_size, self.token_size - self.token_overlap)  # (B, num_patches, token_size)
            embedded = self.embed_layers[d](patches)  # (B, num_patches, d_model)
            x_d = self.pos_encoders[d](embedded)

            for layer in self.transformer_layers[d]:
                x_d = layer(x_d)
                self.attn_weights[d] = layer.attn_weights  # optional

            x_d = x_d.reshape(B, -1)  # (B, num_patches * d_model)
            out_d = self.reconstructors[d](x_d)  # (B, seq_len)
            outputs.append(out_d.unsqueeze(1))

        # Combine across variates: (B, D, seq_len)
        return torch.cat(outputs, dim=1)

# Train independent model

model_ind = IndependentTransformerEncoder().to(device)
criterion = MaskedWeightedMSELoss()
optimizer_ind = torch.optim.Adam(model_ind.parameters(), lr=1e-3)
scheduler_ind = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ind, T_max=epoch_num)

ind_output_path = output_path + "ind/"
os.makedirs(ind_output_path, exist_ok=True)

# best_val_loss = float('inf')
# early_stop_counter = 0
# train_losses, val_losses = [], []

# for epoch in range(epoch_num):
#     model_ind.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)
        
#         # Transpose to (B, D, L)
#         gt = gt.transpose(1, 2)
#         masked = masked.transpose(1, 2)
#         mask = mask.transpose(1, 2)

#         optimizer_ind.zero_grad()
#         out = model_ind(masked, mask)  # (B, D, L)
#         loss = criterion(out, gt, mask, args.loss_r)
#         loss.backward()
#         optimizer_ind.step()
#         train_loss += loss.item()

#     model_ind.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#             gt = gt.transpose(1, 2)
#             masked = masked.transpose(1, 2)
#             mask = mask.transpose(1, 2)

#             out = model_ind(masked, mask)
#             loss = criterion(out, gt, mask, args.loss_r)
#             val_loss += loss.item()

#     train_losses.append(train_loss / len(train_loader))
#     val_losses.append(val_loss / len(val_loader))
#     print(f"[Ind] Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

#     scheduler_ind.step()

#     if val_losses[-1] < best_val_loss:
#         best_val_loss = val_losses[-1]
#         early_stop_counter = 0
#         torch.save(model_ind.state_dict(), ind_output_path + "best_model.pth")
#         last_attn_weights = model_ind.attn_weights
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[Ind] Early stopping")
#             break

# Test independent model

model_ind.load_state_dict(torch.load(ind_output_path + "best_model.pth"))
model_ind.eval()
mse_total, count = 0, 0

for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)
    gt = gt.transpose(1, 2)
    masked = masked.transpose(1, 2)
    mask = mask.transpose(1, 2)

    with torch.no_grad():
        out = model_ind(masked, mask)  # (B, D, L)

    gt_np = gt[0].cpu().numpy()       # (D, L)
    out_np = out[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    # gt_np = gt_np.transpose(0, 1)
    # out_np = out_np.transpose(0, 1)
    # mask_np = mask_np.transpose(0, 1)

    # gt_denorm = denormalize(gt_np, data_std, data_mean)
    # out_denorm = denormalize(out_np, data_std, data_mean)

    if args.missing_rate != 0:
        mse_total += ((out_np[~mask_np] - gt_np[~mask_np]) ** 2).sum()
        count += (~mask_np).sum()
    else:
        mse_total += ((out_np - gt_np) ** 2).sum()
        count += mask_np.sum()

    # if i < 2:
    #     for d in range(gt_np.shape[0]):
    #         plt.figure()
    #         plt.plot(gt_denorm[d], label="GT")
    #         plt.plot(out_denorm[d], label="Output")
    #         plt.plot(mask_np[d] * max(gt_denorm[d]), label="Mask")
    #         plt.legend()
    #         plt.title(f"Test Case {i}, Variate {d}")
    #         plt.savefig(f"{ind_output_path}test_case_{i}_variate_{d}.png")

print("[Ind] Masked MSE on test set:", mse_total / count)
print("[Ind] Working directory:", os.getcwd())
with open("test_results.txt", "a") as log_file:
    log_file.write(output_path + "\n")
    log_file.write("[Ind] Masked MSE on test set:" + str(mse_total / count) + "\n")




# Train baseline model
baseline_model = BaselineTransformerEncoder(token_t_size=args.seq_len, token_t_overlap=0).to(device)
criterion = MaskedWeightedMSELoss()
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=epoch_num)

baseline_output_path = output_path + "baseline/"
os.makedirs(baseline_output_path, exist_ok=True)

best_val_loss_b = float('inf')
train_losses_b, val_losses_b = [], []
early_stop_counter = 0

for epoch in range(epoch_num):
    baseline_model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        baseline_optimizer.zero_grad()
        out = baseline_model(masked, mask)
        recon = reconstruct_from_vertical_patches(temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
                                                  token_t_size=args.seq_len,
                                                  token_t_overlap=0)
        # recon = out
        loss = criterion(recon, gt, mask, args.loss_r)
        loss.backward()
        baseline_optimizer.step()
        train_loss += loss.item()

    baseline_model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = baseline_model(masked, mask)
            recon = reconstruct_from_vertical_patches(temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
                                                      token_t_size=args.seq_len,
                                                      token_t_overlap=0)
            # recon = out
            loss = criterion(recon, gt, mask, args.loss_r)
            val_loss += loss.item()

    train_losses_b.append(train_loss / len(train_loader))
    val_losses_b.append(val_loss / len(val_loader))
    print(f"[Baseline] Epoch {epoch}, Train Loss: {train_losses_b[-1]:.4f}, Val Loss: {val_losses_b[-1]:.4f}")

    baseline_scheduler.step()

    if val_losses_b[-1] < best_val_loss_b:
        best_val_loss_b = val_losses_b[-1]
        early_stop_counter = 0
        torch.save(baseline_model.state_dict(), f"{baseline_output_path}/best_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("[Baseline] Early stopping")
            break

plt.figure()
plt.plot(train_losses_b, label="Train")
plt.plot(val_losses_b, label="Val")
plt.legend()
plt.title("Baseline Loss Curve")
plt.savefig(f"{baseline_output_path}/loss_curve.png")



# Load best baseline model
baseline_model.load_state_dict(torch.load(f"{baseline_output_path}/best_model.pth"))
baseline_model.eval()

mse_total_b, count_b = 0.0, 0
for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
    with torch.no_grad():
        out = baseline_model(masked, mask)
        out = reconstruct_from_vertical_patches(
            temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
            token_t_size=args.seq_len,
            token_t_overlap=0
        )

    gt_np = gt.cpu().numpy().squeeze()
    out_np = out.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()

    if gt_np.ndim == 1: gt_np = gt_np[np.newaxis, :]
    if out_np.ndim == 1: out_np = out_np[np.newaxis, :]
    if mask_np.ndim == 1: mask_np = mask_np[np.newaxis, :]

    gt_denorm = denormalize(gt_np, data_mean, data_std)
    out_denorm = denormalize(out_np, data_mean, data_std)

    mse = ((gt_np[~mask_np] - out_np[~mask_np])**2).sum()
    mse_total_b += mse
    count_b += (~mask_np).sum()

    if i < 2:
        plt.figure()
        plt.plot(gt_denorm[:, 0], label="GT")
        plt.plot(out_denorm[:, 0], label="Output")
        plt.plot(mask_np[:, 0] * np.max(gt_denorm[:, 0]), label="Mask")
        plt.legend()
        plt.title(f"Baseline Test Case {i}")
        plt.savefig(f"{baseline_output_path}/test_case_{i}.png")

print("[Baseline] Masked MSE on test set:", mse_total_b / count_b)
with open("test_results.txt", "a") as log_file:
    print("Writing to file...")
    log_file.write(output_path + "\n")
    log_file.write("[Baseline] Masked MSE on test set:" + str(mse_total_b / count_b) + "\n")

# Plot attention map
for i in range(len(baseline_model.layers)):
    if hasattr(baseline_model.layers[i], 'attn_weights') and baseline_model.layers[i].attn_weights is not None:
        attn_map = baseline_model.layers[-1].attn_weights[0].cpu().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title("Attention Map Layer {}".format(i))
        plt.savefig(f"{baseline_output_path}/attention_map_layer{i}.png")





# # Train multi-token model
# multi_model = BaselineTransformerEncoder().to(device)
# multi_optimizer = torch.optim.Adam(multi_model.parameters(), lr=1e-3)
# multi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(multi_optimizer, T_max=epoch_num)

# multi_output_path = output_path + "multi/"
# os.makedirs(multi_output_path, exist_ok=True)

# best_val_loss_multi = float('inf')
# train_losses_multi, val_losses_multi = [], []
# early_stop_counter = 0

# for epoch in range(epoch_num):
#     multi_model.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#         multi_optimizer.zero_grad()
#         out = multi_model(masked, mask)
#         recon = reconstruct_from_vertical_patches(temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#                                                   token_t_size=args.token_t_size,
#                                                   token_t_overlap=args.token_t_overlap)
#         loss = criterion(recon, gt, mask, args.loss_r)
#         loss.backward()
#         multi_optimizer.step()
#         train_loss += loss.item()

#     multi_model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#             out = multi_model(masked, mask)
#             recon = reconstruct_from_vertical_patches(temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#                                                       token_t_size=args.token_t_size,
#                                                       token_t_overlap=args.token_t_overlap)
#             loss = criterion(recon, gt, mask, args.loss_r)
#             val_loss += loss.item()

#     train_losses_multi.append(train_loss / len(train_loader))
#     val_losses_multi.append(val_loss / len(val_loader))
#     print(f"[Multi] Epoch {epoch}, Train Loss: {train_losses_multi[-1]:.4f}, Val Loss: {val_losses_multi[-1]:.4f}")

#     multi_scheduler.step()

#     if val_losses_multi[-1] < best_val_loss_multi:
#         best_val_loss_multi = val_losses_multi[-1]
#         early_stop_counter = 0
#         torch.save(multi_model.state_dict(), f"{multi_output_path}/best_model.pth")
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[Multi] Early stopping")
#             break

# plt.figure()
# plt.plot(train_losses_multi, label="Train")
# plt.plot(val_losses_multi, label="Val")
# plt.legend()
# plt.title("Loss Curve")
# plt.savefig(f"{multi_output_path}/loss_curve.png")



# # Load best multi model
# multi_model.load_state_dict(torch.load(f"{multi_output_path}/best_model.pth"))
# multi_model.eval()

# mse_total_multi, count_multi = 0.0, 0
# for i, (gt, masked, mask) in enumerate(test_loader):
#     gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#     with torch.no_grad():
#         out = multi_model(masked, mask)
#         out = reconstruct_from_vertical_patches(
#             temporal_tokens=out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#             token_t_size=args.token_t_size,
#             token_t_overlap=args.token_t_overlap
#         )

#     gt_np = gt.cpu().numpy().squeeze()
#     out_np = out.cpu().numpy().squeeze()
#     mask_np = mask.cpu().numpy().squeeze()

#     if gt_np.ndim == 1: gt_np = gt_np[np.newaxis, :]
#     if out_np.ndim == 1: out_np = out_np[np.newaxis, :]
#     if mask_np.ndim == 1: mask_np = mask_np[np.newaxis, :]

#     gt_denorm = denormalize(gt_np, data_mean, data_std)
#     out_denorm = denormalize(out_np, data_mean, data_std)

#     mse = ((gt_np[~mask_np] - out_np[~mask_np])**2).sum()
#     mse_total_multi += mse
#     count_multi += (~mask_np).sum()

#     if i < 2:
#         plt.figure()
#         plt.plot(gt_denorm[:, 0], label="GT")
#         plt.plot(out_denorm[:, 0], label="Output")
#         plt.plot(mask_np[:, 0] * np.max(gt_denorm[:, 0]), label="Mask")
#         plt.legend()
#         plt.title(f"Test Case {i}")
#         plt.savefig(f"{multi_output_path}/test_case_{i}.png")

# print("[Multi] Masked MSE on test set:", mse_total_multi / count_multi)
# with open("test_results.txt", "a") as log_file:
#     print("Writing to file...")
#     log_file.write(output_path + "\n")
#     log_file.write("[Multi] Masked MSE on test set:" + str(mse_total_multi / count_multi) + "\n")

# # Plot attention map
# for i in range(len(multi_model.layers)):
#     if hasattr(multi_model.layers[i], 'attn_weights') and multi_model.layers[i].attn_weights is not None:
#         attn_map = multi_model.layers[-1].attn_weights[0].cpu().numpy()
#         plt.figure(figsize=(6, 5))
#         plt.imshow(attn_map, cmap='viridis')
#         plt.colorbar()
#         plt.title("Attention Map Layer {}".format(i))
#         plt.savefig(f"{multi_output_path}/attention_map_layer{i}.png")




# 2D Model initialization
model = TransformerEncoder().to(device)
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
        out = model(masked, mask)
        recon = reconstruct_from_patches(
            out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
            token_t_size=args.token_t_size,
            token_t_overlap=args.token_t_overlap,
            token_d_size=args.token_d_size,
            token_d_overlap=args.token_d_overlap
        )
        loss = criterion(recon, gt, mask, args.loss_r)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = model(masked, mask)
            recon = reconstruct_from_patches(
                out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
                token_t_size=args.token_t_size,
                token_t_overlap=args.token_t_overlap,
                token_d_size=args.token_d_size,
                token_d_overlap=args.token_d_overlap
            )
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
        out = reconstruct_from_patches(
            out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
            token_t_size=args.token_t_size,
            token_t_overlap=args.token_t_overlap,
            token_d_size=args.token_d_size,
            token_d_overlap=args.token_d_overlap
        )

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
        plt.figure()
        plt.plot(gt_denorm[:, 0], label="GT")
        plt.plot(out_denorm[:, 0], label="Output")
        plt.plot(mask_np[:, 0] * np.max(gt_denorm[:, 0]), label="Mask")
        plt.legend()
        plt.title(f"Test Case {i}")
        plt.savefig(f"{twoD_output_path}/test_case_{i}.png")

print("[2D] Masked MSE on test set:", mse_total / count)
with open("test_results.txt", "a") as log_file:
    print("[2D] Writing to file...")
    log_file.write("[2D] Masked MSE on test set:" + str(mse_total / count) + "\n" + "\n" + "\n")


# Plot attention map
for i in range(len(model.layers)):
    if hasattr(model.layers[i], 'attn_weights') and model.layers[i].attn_weights is not None:
        attn_map = model.layers[-1].attn_weights[0].cpu().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title("Attention Map Layer {}".format(i))
        plt.savefig(f"{twoD_output_path}/attention_map_layer{i}.png")





# # Training the 1D mixture model
# mix_model = MixedTokenTransformer(D=D).to(device)
# mix_optimizer = torch.optim.Adam(mix_model.parameters(), lr=1e-3)
# mix_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mix_optimizer, T_max=epoch_num)

# mix_output_path = output_path + "mix/"
# os.makedirs(mix_output_path, exist_ok=True)

# best_val_loss_b = float('inf')
# train_losses_b, val_losses_b = [], []
# early_stop_counter = 0

# for epoch in range(epoch_num):
#     mix_model.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#         mix_optimizer.zero_grad()
#         temporal_tokens, feature_tokens = mix_model(masked, mask)
#         recon = reconstruct_from_mixed_tokens(device = device, temporal_tokens = temporal_tokens, feature_tokens = feature_tokens, 
#                                               B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#                                                   token_t_size=args.token_t_size,
#                                                   token_t_overlap=args.token_t_overlap)
#         loss = criterion(recon, gt, mask, args.loss_r)
#         loss.backward()
#         mix_optimizer.step()
#         train_loss += loss.item()

#     mix_model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#             out = mix_model(masked, mask)
#             recon = reconstruct_from_mixed_tokens(device = device, temporal_tokens = temporal_tokens, feature_tokens = feature_tokens,
#                                                 B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#                                                 token_t_size=args.token_t_size,
#                                                 token_t_overlap=args.token_t_overlap)
#             loss = criterion(recon, gt, mask, args.loss_r)
#             val_loss += loss.item()

#     train_losses_b.append(train_loss / len(train_loader))
#     val_losses_b.append(val_loss / len(val_loader))
#     print(f"[Mix] Epoch {epoch}, Train Loss: {train_losses_b[-1]:.4f}, Val Loss: {val_losses_b[-1]:.4f}")

#     mix_scheduler.step()

#     if val_losses_b[-1] < best_val_loss_b:
#         best_val_loss_b = val_losses_b[-1]
#         early_stop_counter = 0
#         torch.save(mix_model.state_dict(), f"{mix_output_path}/best_model.pth")
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[Mix] Early stopping")
#             break

# plt.figure()
# plt.plot(train_losses_b, label="Train")
# plt.plot(val_losses_b, label="Val")
# plt.legend()
# plt.title("Mix Model Loss Curve")
# plt.savefig(f"{mix_output_path}/loss_curve.png")




# mix_model.load_state_dict(torch.load(f"{mix_output_path}/best_model.pth"))
# mix_model.eval()
# mse_total_a, count_a = 0.0, 0

# for i, (gt, masked, mask) in enumerate(test_loader):
#     gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#     with torch.no_grad():
#         out = mix_model(masked, mask)
#         out = reconstruct_from_mixed_tokens(device = device, temporal_tokens = temporal_tokens, feature_tokens = feature_tokens,
#                                             B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#                                             token_t_size=args.token_t_size,
#                                             token_t_overlap=args.token_t_overlap)  
        
#         gt_np = gt.cpu().numpy().squeeze()
#         out_np = out.cpu().numpy().squeeze()
#         mask_np = mask.cpu().numpy().squeeze()

#     gt_denorm = denormalize(gt_np, data_mean, data_std)
#     recon_denorm = denormalize(out_np, data_mean, data_std)
#     mse = ((out_np[~mask_np] - gt_np[~mask_np])**2).sum().item()
#     mse_total_a += mse
#     count_a += (~mask_np).sum().item()

#     if i < 2:
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(gt_np[:, 0], label="GT")
#         plt.plot(out_np[:, 0], label="Output")
#         plt.plot(mask_np[:, 0] * np.max(gt_np[:, 0]), label="Mask")
#         plt.legend()
#         plt.title(f"Mixed Test Case {i}")
#         plt.savefig(f"{mix_output_path}/mixed_test_case_{i}.png")

# print("[Mixed] Masked MSE on test set:", mse_total_a / count_a)
# with open("test_results.txt", "a") as log_file:
#     print("Writing to file...")
#     log_file.write("[Mixed] Masked MSE on test set:" + str(mse_total_a / count_a) + "\n")



