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
# dt_seconds = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 's')
# sampling_rate = 1 / dt_seconds

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

class twoDTransformerEncoder(nn.Module):
    def __init__(self, args=args):
        super().__init__()
        # Patch2DEmbedding input: (B, T, D)
        # Output: (B, N_patches, d_model)
        self.args = args
        d_model = args.token_t_size * args.nhead  # d_model = token_t_size * nhead
        self.embed = Patch2DEmbedding(
            args.token_t_size, args.token_t_overlap,
            args.token_d_size, args.token_d_overlap,
            d_model=d_model
        )

        # PositionalEncoding input: (B, N_patches, d_model)
        # Output: (B, N_patches, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer layers process (B, N_patches, d_model)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, args.nhead)
            for _ in range(args.num_layers)
        ])

        # Final projection maps (B, N_patches, d_model) → (B, N_patches, patch_size)
        # where patch_size = token_t_size * token_d_size
        self.project = nn.Linear(d_model, args.token_t_size * args.token_d_size)

    def forward(self, x, mask):
        B, T, D = x.shape
        # Input: x, mask → (B, T, D)
        x = torch.where(mask, x, 0.0)  # (B, T, D)

        x_embed = self.embed(x)  # → (B, N_patches, d_model)
        x_embed = self.pos_encoder(x_embed)  # → (B, N_patches, d_model)

        for layer in self.layers:
            x_embed = layer(x_embed)  # remains (B, N_patches, d_model)

        out = self.project(x_embed)  # → (B, N_patches, patch_size)
        out = reconstruct_from_patches(
            out, B=B, T=T, D=D,
            token_t_size=self.args.token_t_size,
            token_t_overlap=self.args.token_t_overlap,
            token_d_size=self.args.token_d_size,
            token_d_overlap=self.args.token_d_overlap
        )
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
    

# Multiscale independent imputation

class IndependentPatchEmbedding(nn.Module):
    def __init__(self, token_size, d_model):
        super().__init__()
        self.linear = nn.Linear(token_size, d_model)

    def forward(self, x):
        return self.linear(x)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_sizes = [int(args.seq_len / 16), int(args.seq_len / 8), int(args.seq_len / 4), int(args.seq_len / 2)]
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.seq_len = args.seq_len
        self.token_overlap = args.token_t_overlap

        self.mask_token = nn.Parameter(torch.randn(1))

        self.branches = nn.ModuleList()
        self.num_patches_list = []

        for token_size in self.token_sizes:
            d_model = token_size * args.nhead  # d_model should be a multiple of nhead
            branch = nn.ModuleDict({
                'embed': IndependentPatchEmbedding(token_size, d_model),
                'pos_encoder': PositionalEncoding(d_model=d_model),
                'layers': nn.ModuleList([CustomTransformerEncoderLayer(d_model, args.nhead) for _ in range(self.num_layers)]),
                'reconstructor': nn.Linear(d_model * (((args.seq_len - token_size) // (token_size - self.token_overlap)) + 1), args.seq_len)
            })
            self.num_patches_list.append(((args.seq_len - token_size) // (token_size - self.token_overlap)) + 1)
            self.branches.append(branch)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.ones(len(self.token_sizes)))  # raw weights, softmax later

    def forward(self, x, mask=None):
        B, L = x.shape
        x_masked = torch.where(mask, x, self.mask_token.expand_as(x))
        # x_masked=x

        outputs = []
        for i, token_size in enumerate(self.token_sizes):
            branch = self.branches[i]
            stride = token_size - self.token_overlap

            # Patchify
            patches = x_masked.unfold(1, token_size, stride)  # (B, num_patches, token_size)

            # Embedding + Positional Encoding
            x_embed = branch['embed'](patches)
            x_pos = branch['pos_encoder'](x_embed)

            # Transformer Layers
            x_trans = x_pos
            for layer in branch['layers']:
                x_trans = layer(x_trans)
            
            # Flatten and reconstruct
            x_flat = x_trans.reshape(B, -1)
            out = branch['reconstructor'](x_flat)  # (B, seq_len)
            outputs.append(out)

        # Combine outputs using softmax-weighted sum
        weights = torch.softmax(self.alpha, dim=0)  # shape (4,)
        out_combined = sum(w * o for w, o in zip(weights, outputs))

        # return out_combined, weights, outputs
        return out_combined, weights
    
class MultiscaleIndependentTransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.D = args.token_d_size
        self.seq_len = args.seq_len
        self.mask_token = nn.Parameter(torch.randn(1))

        # Create one multiscale TransformerEncoder per variate
        self.branch_models = nn.ModuleList([
            TransformerEncoder(args) for _ in range(self.D)
        ])

    def forward(self, x, mask=None):
        # x: (B, D, L), mask: (B, D, L)
        B, D, L = x.shape
        outputs = []
        all_weights = []

        for d in range(self.D):
            x_d = x[:, d, :]                   # (B, L)
            mask_d = mask[:, d, :] if mask is not None else torch.ones_like(x_d, dtype=torch.bool)
            x_d_masked = torch.where(mask_d, x_d, self.mask_token.expand_as(x_d))

            # Forward through the d-th branch
            out_d, weights_d = self.branch_models[d](x_d_masked, mask_d)
            outputs.append(out_d.unsqueeze(1))  # (B, 1, L)
            all_weights.append(weights_d.unsqueeze(1))  # (1, 4) -> (B, 1, 4)

        # Stack outputs: (B, D, L=seq_len)
        imputed = torch.cat(outputs, dim=1)
        # Stack weights: (B, D, 4)
        weights = torch.cat(all_weights, dim=1)

        return imputed

    


# Independently imputation



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
    
class FusionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x_indep, x_2d):
        # x: (B, D, L) → (B*L, D, d_model) for attention
        B, D, L = x_indep.shape
        x_indep = x_indep.permute(0, 2, 1)  # (B, L, D)
        x_2d = x_2d.permute(0, 2, 1)  # (B, L, D)
        fused, _ = self.cross_attn(x_indep, x_2d, x_2d)
        combined = torch.cat([x_indep, fused], dim=-1)
        return self.linear(combined).permute(0, 2, 1)  # (B, D, L)

    
class HybridImputer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.indep_encoder = MultiscaleIndependentTransformerEncoder(args)
        self.twod_encoder = twoDTransformerEncoder()
        self.fusion_type = "cross_attn"  # "add", "concat", "cross_attn"
        if self.fusion_type == "cross_attn":
            self.fusion = FusionBlock(d_model=args.seq_len)

    def forward(self, x, mask):
        out_indep = self.indep_encoder(x.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
        patch_out = self.twod_encoder(x, mask)  # (B, N_patches, patch_size)
        # print("patch_out shape:", patch_out.shape)  # (B, L, D)
        out_2d = reconstruct_from_patches(
            patch_out, B=mask.shape[0], T=mask.shape[1], D=mask.shape[2],
            token_t_size=args.token_t_size,
            token_t_overlap=args.token_t_overlap,
            token_d_size=args.token_d_size,
            token_d_overlap=args.token_d_overlap
        )

        if self.fusion_type == "add":
            return (out_indep + out_2d) / 2
        elif self.fusion_type == "concat":
            return torch.cat([out_indep, out_2d], dim=1)  # or reduce to (B, D, L) later
        elif self.fusion_type == "cross_attn":
            return self.fusion(out_indep, out_2d)
        
# Train and test the hybrid model     
# model_hyb = HybridImputer(args).to(device)
# criterion = MaskedWeightedMSELoss()
# optimizer_hyb = torch.optim.Adam(model_hyb.parameters(), lr=1e-3)
# scheduler_hyb = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyb, T_max=epoch_num)

# hyb_output_path = output_path + "hybrid/"
# os.makedirs(hyb_output_path, exist_ok=True)

# best_val_loss = float('inf')
# early_stop_counter = 0
# train_losses, val_losses = [], []

# for epoch in range(epoch_num):
#     model_hyb.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)
        

#         optimizer_hyb.zero_grad()
#         out = model_hyb(masked, mask)  # (B, D, L)
#         loss = criterion(out, gt, mask, args.loss_r)
#         loss.backward()
#         optimizer_hyb.step()
#         train_loss += loss.item()

#     model_hyb.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)

#             out = model_hyb(masked, mask)
#             loss = criterion(out, gt, mask, args.loss_r)
#             val_loss += loss.item()

#     train_losses.append(train_loss / len(train_loader))
#     val_losses.append(val_loss / len(val_loader))
#     print(f"[Hybrid] Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

#     scheduler_hyb.step()

#     if val_losses[-1] < best_val_loss:
#         best_val_loss = val_losses[-1]
#         early_stop_counter = 0
#         torch.save(model_hyb.state_dict(), hyb_output_path + "best_model.pth")
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[Hybrid] Early stopping")
#             break

# # Test hybrid model
# model_hyb.load_state_dict(torch.load(hyb_output_path + "best_model.pth"))
# model_hyb.eval()
# mse_total, count = 0, 0

# for i, (gt, masked, mask) in enumerate(test_loader):
#     gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)

#     with torch.no_grad():
#         out = model_hyb(masked, mask)  # (B, D, L)

#     gt_np = gt.cpu().numpy().squeeze()
#     out_np = out.cpu().numpy().squeeze()
#     mask_np = mask.cpu().numpy().squeeze()

#     if gt_np.ndim == 1: gt_np = gt_np[np.newaxis, :]
#     if out_np.ndim == 1: out_np = out_np[np.newaxis, :]
#     if mask_np.ndim == 1: mask_np = mask_np[np.newaxis, :]

#     gt_denorm = denormalize(gt_np, data_mean, data_std)
#     out_denorm = denormalize(out_np, data_mean, data_std)


#     if i < 2:
#         os.makedirs(f"{hyb_output_path}/test_case_{i}/", exist_ok=True)
#         for j in range (0, D):
#             plt.figure()
#             plt.plot(gt_denorm[0, :, j], label="GT")
#             plt.plot(out_denorm[0, :, j], label="Output")
#             plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
#             plt.legend()
#             plt.title(f"Hybrid Test Case {i} column {j}")
#             plt.savefig(f"{hyb_output_path}/test_case_{i}/column_{j}.png")


#     if args.missing_rate != 0:
#         mse_total += ((out_np[~mask_np] - gt_np[~mask_np]) ** 2).sum()
#         count += (~mask_np).sum()
#     else:
#         mse_total += ((out_np - gt_np) ** 2).sum()
#         count += mask_np.sum()


# print("[Hybrid] Masked MSE on test set:", mse_total / count)
# print("[Hybrid] Working directory:", os.getcwd())
# with open("test_results.txt", "a") as log_file:
#     log_file.write(output_path + "\n")
#     log_file.write("[Hybrid] Masked MSE on test set:" + str(mse_total / count) + "\n")



# Train independent model

# model_ind = MultiscaleIndependentTransformerEncoder(args).to(device)
# criterion = MaskedWeightedMSELoss()
# optimizer_ind = torch.optim.Adam(model_ind.parameters(), lr=1e-3)
# scheduler_ind = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ind, T_max=epoch_num)

# ind_output_path = output_path + "ind/"
# os.makedirs(ind_output_path, exist_ok=True)

# best_val_loss = float('inf')
# early_stop_counter = 0
# train_losses, val_losses = [], []

# for epoch in range(epoch_num):
#     model_ind.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)
        
#         # Transpose to (B, D, L)
#         # gt = gt.transpose(1, 2)
#         # masked = masked.transpose(1, 2)
#         # mask = mask.transpose(1, 2)

#         optimizer_ind.zero_grad()
#         out = model_ind(masked.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
#         loss = criterion(out, gt, mask, args.loss_r)
#         loss.backward()
#         optimizer_ind.step()
#         train_loss += loss.item()

#     model_ind.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#             # gt = gt.transpose(1, 2)
#             # masked = masked.transpose(1, 2)
#             # mask = mask.transpose(1, 2)

#             out = model_ind(masked.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
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
#         # last_attn_weights = model_ind.attn_weights
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[Ind] Early stopping")
#             break

# # Test independent model

# model_ind.load_state_dict(torch.load(ind_output_path + "best_model.pth"))
# model_ind.eval()
# mse_total, count = 0, 0

# for i, (gt, masked, mask) in enumerate(test_loader):
#     gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)  # (B, L, D)
#     # gt = gt.transpose(1, 2)
#     # masked = masked.transpose(1, 2)
#     # mask = mask.transpose(1, 2)

#     with torch.no_grad():
#         out = model_ind(masked.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)  # (B, L, D)

#     gt_np = gt.cpu().numpy().squeeze()
#     out_np = out.cpu().numpy().squeeze()
#     mask_np = mask.cpu().numpy().squeeze()

#     if gt_np.ndim == 1: gt_np = gt_np[np.newaxis, :]
#     if out_np.ndim == 1: out_np = out_np[np.newaxis, :]
#     if mask_np.ndim == 1: mask_np = mask_np[np.newaxis, :]

#     gt_denorm = denormalize(gt_np, data_mean, data_std)
#     out_denorm = denormalize(out_np, data_mean, data_std)

#     if i < 2:
#         os.makedirs(f"{ind_output_path}/test_case_{i}/", exist_ok=True)
#         for j in range (0, D):
#             plt.figure()
#             plt.plot(gt_denorm[0, :, j], label="GT")
#             plt.plot(out_denorm[0, :, j], label="Output")
#             plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
#             plt.legend()
#             plt.title(f"Independent Test Case {i} column {j}")
#             plt.savefig(f"{ind_output_path}/test_case_{i}/column_{j}.png")

#     if args.missing_rate != 0:
#         mse_total += ((out_np[~mask_np] - gt_np[~mask_np]) ** 2).sum()
#         count += (~mask_np).sum()
#     else:
#         mse_total += ((out_np - gt_np) ** 2).sum()
#         count += mask_np.sum()

    

# print("[Ind] Masked MSE on test set:", mse_total / count)
# print("[Ind] Working directory:", os.getcwd())
# with open("test_results.txt", "a") as log_file:
#     log_file.write(output_path + "\n")
#     log_file.write("[Ind] Masked MSE on test set:" + str(mse_total / count) + "\n")



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
        os.makedirs(f"{baseline_output_path}/test_case_{i}/", exist_ok=True)
        for j in range (0, D):
            plt.figure()
            plt.plot(gt_denorm[0, :, j], label="GT")
            plt.plot(out_denorm[0, :, j], label="Output")
            plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
            plt.legend()
            plt.title(f"Baseline Test Case {i} column {j}")
            plt.savefig(f"{baseline_output_path}/test_case_{i}/column_{j}.png")

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







# 2D Model initialization
# model = twoDTransformerEncoder().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

# twoD_output_path = output_path + "2D/"
# os.makedirs(twoD_output_path, exist_ok=True)

# # Training and Validation
# best_val_loss = float('inf')
# early_stop_counter = 0
# train_losses, val_losses = [], []

# for epoch in range(epoch_num):
#     model.train()
#     train_loss = 0
#     for gt, masked, mask in train_loader:
#         gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#         optimizer.zero_grad()
#         recon = model(masked, mask)
#         # recon = reconstruct_from_patches(
#         #     out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#         #     token_t_size=args.token_t_size,
#         #     token_t_overlap=args.token_t_overlap,
#         #     token_d_size=args.token_d_size,
#         #     token_d_overlap=args.token_d_overlap
#         # )
#         loss = criterion(recon, gt, mask, args.loss_r)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for gt, masked, mask in val_loader:
#             gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#             recon = model(masked, mask)
#             # recon = reconstruct_from_patches(
#             #     out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#             #     token_t_size=args.token_t_size,
#             #     token_t_overlap=args.token_t_overlap,
#             #     token_d_size=args.token_d_size,
#             #     token_d_overlap=args.token_d_overlap
#             # )
#             loss = criterion(recon, gt, mask, args.loss_r)
#             val_loss += loss.item()

#     train_losses.append(train_loss / len(train_loader))
#     val_losses.append(val_loss / len(val_loader))
#     print(f"[2D] Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

#     scheduler.step()

#     if val_losses[-1] < best_val_loss:
#         best_val_loss = val_losses[-1]
#         early_stop_counter = 0
#         torch.save(model.state_dict(), f"{twoD_output_path}/best_model.pth")
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= 5:
#             print("[2D] Early stopping")
#             break

# # Plot loss curve
# plt.figure()
# plt.plot(train_losses, label="Train")
# plt.plot(val_losses, label="Val")
# plt.legend()
# plt.title("Loss Curve")
# plt.savefig(f"{twoD_output_path}/loss_curve.png")



# # Load best model for evaluation
# model.load_state_dict(torch.load(f"{twoD_output_path}/best_model.pth"))
# model.eval()

# # Evaluation
# mse_total, count = 0.0, 0
# for i, (gt, masked, mask) in enumerate(test_loader):
#     gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
#     with torch.no_grad():
#         out = model(masked, mask)
#         # out = reconstruct_from_patches(
#         #     out, B=gt.shape[0], T=gt.shape[1], D=gt.shape[2],
#         #     token_t_size=args.token_t_size,
#         #     token_t_overlap=args.token_t_overlap,
#         #     token_d_size=args.token_d_size,
#         #     token_d_overlap=args.token_d_overlap
#         # )

#     gt_np = gt.cpu().numpy().squeeze()
#     if gt_np.ndim == 1:
#         gt_np = gt_np[np.newaxis, :]
#     out_np = out.cpu().numpy().squeeze()
#     if out_np.ndim == 1:
#         out_np = out_np[np.newaxis, :]
#     mask_np = mask.cpu().numpy().squeeze()
#     if mask_np.ndim == 1:
#         mask_np = mask_np[np.newaxis, :]


#     gt_denorm = denormalize(gt_np, data_mean, data_std)
#     out_denorm = denormalize(out_np, data_mean, data_std)

#     mse = ((gt_np[~mask_np] - out_np[~mask_np])**2).sum()
#     mse_total += mse
#     count += (~mask_np).sum()

#     if i < 2:
#         os.makedirs(f"{twoD_output_path}/test_case_{i}/", exist_ok=True)
#         for j in range (0, D):
#             plt.figure()
#             plt.plot(gt_denorm[0, :, j], label="GT")
#             plt.plot(out_denorm[0, :, j], label="Output")
#             plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
#             plt.legend()
#             plt.title(f"Baseline Test Case {i} column {j}")
#             plt.savefig(f"{twoD_output_path}/test_case_{i}/column_{j}.png")

# print("[2D] Masked MSE on test set:", mse_total / count)
# with open("test_results.txt", "a") as log_file:
#     print("[2D] Writing to file...")
#     log_file.write("[2D] Masked MSE on test set:" + str(mse_total / count) + "\n" + "\n" + "\n")


# # Plot attention map
# for i in range(len(model.layers)):
#     if hasattr(model.layers[i], 'attn_weights') and model.layers[i].attn_weights is not None:
#         attn_map = model.layers[-1].attn_weights[0].cpu().numpy()
#         plt.figure(figsize=(6, 5))
#         plt.imshow(attn_map, cmap='viridis')
#         plt.colorbar()
#         plt.title("Attention Map Layer {}".format(i))
#         plt.savefig(f"{twoD_output_path}/attention_map_layer{i}.png")







