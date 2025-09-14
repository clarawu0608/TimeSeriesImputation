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

class FFNBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ffn(x))

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
    
class CustomCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, query, context, attn_mask=None):
        """
        query: (B, N_q, d_model) – the sequence that will receive the attended output.
        context: (B, N_kv, d_model) – the source sequence that provides keys and values.
        """
        attn_output, attn_weights = self.cross_attn(query, context, context, attn_mask=attn_mask, need_weights=True)
        self.attn_weights = attn_weights.detach()

        query = self.norm1(query + self.dropout1(attn_output))
        query2 = self.linear2(self.dropout(torch.relu(self.linear1(query))))
        return self.norm2(query + self.dropout2(query2))

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

class FullSequenceEmbedding(nn.Module):
    def __init__(self, input_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            Tensor of shape (B, D, d_model)
        """
        x = x.permute(0, 2, 1)  # (B, D, T)
        out = self.proj(x)      # (B, D, d_model)
        return out


class MultiTokenTransformerEncoder(nn.Module):
    def __init__(self, nhead=args.nhead, seq_len = args.seq_len, num_layers=args.num_layers, token_t_size=args.token_t_size, token_t_overlap=args.token_t_overlap):
        super().__init__()
        d_model = token_t_size * nhead
        self.embed1 = TimeOnlyEmbedding(
            token_t_size = token_t_size,
            token_t_overlap = token_t_overlap,
            d_model = token_t_size * nhead
        ) # (B, N_t * D, d_model)
        self.embed2 = FullSequenceEmbedding(
            input_len = seq_len,
            d_model = token_t_size * nhead
        ) # (B, D, d_model)
        self.token_t_size = token_t_size
        self.token_t_overlap = token_t_overlap
        self.pos_encoder = PositionalEncoding(d_model)
        self.intra_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.cross_layers = nn.ModuleList([
            CustomCrossAttentionLayer(d_model, nhead, args.dropout)
            for _ in range(num_layers)
        ])
        self.project = nn.Linear(d_model, token_t_size)
        self.dropout = nn.Dropout(p=args.dropout)
        self.mask_token = nn.Parameter(torch.randn(1))
        self.cross_ffn = FFNBlock(d_model, dim_feedforward=4 * d_model, dropout=args.dropout)


    def forward(self, x, mask):
        B, T, D = x.shape
        x = torch.where(mask, x, self.mask_token.expand_as(x))

        # Patch-level tokens for intra-attention
        patch_embed = self.embed1(x)  # (B, N_t * D, d_model)
        N_t = patch_embed.shape[1] // D
        patch_embed = patch_embed.view(B, D, N_t, -1)  # (B, D, N_t, d_model)
        patch_embed = self.pos_encoder(patch_embed.view(B * D, N_t, -1)).view(B, D, N_t, -1)
        patch_embed = self.dropout(patch_embed)

        # Variate-level full-sequence tokens for cross-attention
        variate_embed = self.embed2(x)  # (B, D, d_model)
        variate_embed = self.pos_encoder(variate_embed)
        variate_embed = self.dropout(variate_embed)

        updated_tokens = []

        for d in range(D):
            # 1. Self-attention on patch tokens of variate d
            tokens_d = patch_embed[:, d]  # (B, N_t, d_model)
            for layer in self.intra_layers:
                tokens_d = tokens_d + layer(tokens_d)  # (B, N_t, d_model)

            # 2. Cross-attention with all variate embeddings (except self if desired)
            q = tokens_d                          # (B, N_t, d_model)
            k = variate_embed                    # (B, D, d_model)
            v = variate_embed                    # (B, D, d_model)
            for layer in self.cross_layers:
                tokens_d = tokens_d + layer(q, k)  # (B, N_t, d_model)

            updated_tokens.append(tokens_d)  # collect (B, N_t, d_model)

        # Concatenate all variate outputs along dim=1 (token axis)
        x_embed = torch.cat(updated_tokens, dim=1)  # (B, N_t * D, d_model)

        # FFN + residual + LayerNorm
        x_embed = self.cross_ffn(x_embed)

        # Project and reconstruct
        out = self.project(x_embed)  # (B, N_t * D, token_t_size)
        recon = reconstruct_from_vertical_patches(
            temporal_tokens=out, B=B, T=T, D=D,
            token_t_size=self.token_t_size,
            token_t_overlap=self.token_t_overlap
        )
        return recon

    
class MultiScaleMultiTokenTransformerEncoder(nn.Module):
    def __init__(self, args = args):
        super().__init__()
        self.token_sizes = [int(args.seq_len / 16), int(args.seq_len / 8), int(args.seq_len / 4), int(args.seq_len / 2)]
        self.token_overlap = args.token_t_overlap
        self.num_layers = args.num_layers
        self.nhead = args.nhead
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        self.D = args.token_d_size  # Number of features / variates

        self.mask_token = nn.Parameter(torch.randn(1))
        self.branches = nn.ModuleList()

        for token_size in self.token_sizes:
            d_model = token_size * self.nhead  # Scale d_model with token_size
            num_patches = ((self.seq_len - token_size) // (token_size - self.token_overlap)) + 1

            branch = nn.ModuleDict({
                'embed': TimeOnlyEmbedding(
                    token_t_size=token_size,
                    token_t_overlap=self.token_overlap,
                    d_model=d_model
                ),
                'pos_encoder': PositionalEncoding(d_model),
                'intra_layers': nn.ModuleList([  
                    CustomTransformerEncoderLayer(d_model, self.nhead)
                    for _ in range(self.num_layers)
                ]),
                'cross_layers': nn.ModuleList([
                    CustomCrossAttentionLayer(d_model, self.nhead, self.dropout)
                    for _ in range(self.num_layers)
                ]),
                'project': nn.Linear(d_model, token_size)  # project back to patch size
            })
            self.branches.append(branch)

        self.alpha = nn.Parameter(torch.ones(len(self.token_sizes)))  # learnable weights for combining outputs

    def forward(self, x, mask):
        B, T, D = x.shape
        x = torch.where(mask, x, self.mask_token.expand_as(x))  # (B, T, D)
        outputs = []

        for idx, token_size in enumerate(self.token_sizes):
            branch = self.branches[idx]
            d_model = token_size * self.nhead

            # 1. Token embedding
            x_embed = branch['embed'](x)  # (B, N_t * D, d_model)
            N_t = x_embed.shape[1] // D

            # 2. Positional encoding
            x_embed = branch['pos_encoder'](x_embed)
            x_embed = x_embed.view(B, D, N_t, d_model).view(B * D, N_t, d_model)

            # 3. Intra-variate attention
            for layer in branch['intra_layers']:
                x_embed = layer(x_embed)

            x_embed = x_embed.view(B, D, N_t, d_model)
            x_embed = x_embed.view(B, D * N_t, d_model)

            # 4. Cross-variate attention
            L = D * N_t
            token_to_variate = torch.arange(L, device=x.device) // N_t
            same_variate = token_to_variate.unsqueeze(0) == token_to_variate.unsqueeze(1)
            attn_mask = same_variate

            for layer in branch['cross_layers']:
                x_embed = layer(x_embed, x_embed, attn_mask=attn_mask)

            # 5. Project back to patch space
            out = branch['project'](x_embed)  # (B, N_t * D, token_size)
            recon = reconstruct_from_vertical_patches(temporal_tokens=out, B=B, T=T, D=D,
                                                  token_t_size=token_size,
                                                  token_t_overlap=args.token_t_overlap) # (B, T, D)
            outputs.append(recon)

            # Combine outputs using softmax-weighted sum
            weights = torch.softmax(self.alpha, dim=0)  # shape (4,)
            out_combined = sum(w * o for w, o in zip(weights, outputs))

        return out_combined  # list of outputs and learned branch weights



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
    

# Train multi-token model
multi_model = MultiTokenTransformerEncoder().to(device)
multi_optimizer = torch.optim.AdamW(multi_model.parameters(), lr=1e-3)
multi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(multi_optimizer, T_max=epoch_num)
criterion = MaskedWeightedMSELoss()

multi_output_path = output_path + "timeXer/"
os.makedirs(multi_output_path, exist_ok=True)

# os.makedirs(f"{multi_output_path}/test/", exist_ok=True)
# for j in range (0, D):
#     plt.figure()
#     plt.plot(data[ :512, j], label="data")
#     plt.plot(mask_matrix[ :512, j] * np.max(data[:, 0]), label="Mask")
#     plt.legend()
#     plt.title(f"Column {j}")
#     plt.savefig(f"{multi_output_path}/test/column_{j}.png")

best_val_loss_multi = float('inf')
train_losses_multi, val_losses_multi = [], []
early_stop_counter = 0

for epoch in range(epoch_num):
    multi_model.train()
    train_loss = 0
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        multi_optimizer.zero_grad()
        out = multi_model(masked, mask)
        loss = criterion(out, gt, mask, args.loss_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(multi_model.parameters(), max_norm=1.0)
        multi_optimizer.step()
        train_loss += loss.item()

    multi_model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = multi_model(masked, mask)
            loss = criterion(out, gt, mask, args.loss_r)
            val_loss += loss.item()

    train_losses_multi.append(train_loss / len(train_loader))
    val_losses_multi.append(val_loss / len(val_loader))
    print(f"[TimeXer] Epoch {epoch}, Train Loss: {train_losses_multi[-1]:.4f}, Val Loss: {val_losses_multi[-1]:.4f}")

    multi_scheduler.step()

    if val_losses_multi[-1] < best_val_loss_multi:
        best_val_loss_multi = val_losses_multi[-1]
        early_stop_counter = 0
        torch.save(multi_model.state_dict(), f"{multi_output_path}/best_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("[TimeXer] Early stopping")
            break

plt.figure()
plt.plot(train_losses_multi, label="Train")
plt.plot(val_losses_multi, label="Val")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{multi_output_path}/loss_curve.png")



# Load best multi model
multi_model.load_state_dict(torch.load(f"{multi_output_path}/best_model.pth"))
multi_model.eval()

mse_total_multi, count_multi = 0.0, 0
for i, (gt, masked, mask) in enumerate(test_loader):
    gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
    with torch.no_grad():
        out = multi_model(masked, mask)


    gt_np = gt.cpu().numpy().squeeze()
    out_np = out.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()

    if gt_np.ndim == 1: gt_np = gt_np[np.newaxis, :]
    if out_np.ndim == 1: out_np = out_np[np.newaxis, :]
    if mask_np.ndim == 1: mask_np = mask_np[np.newaxis, :]

    gt_denorm = denormalize(gt_np, data_mean, data_std)
    out_denorm = denormalize(out_np, data_mean, data_std)

    mse = ((gt_np[~mask_np] - out_np[~mask_np])**2).sum()
    mse_total_multi += mse
    count_multi += (~mask_np).sum()

    if i < 2:
        os.makedirs(f"{multi_output_path}/test_case_{i}/", exist_ok=True)
        for j in range (0, D):
            plt.figure()
            plt.plot(gt_denorm[0, :, j], label="GT")
            plt.plot(out_denorm[0, :, j], label="Output")
            plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask")
            plt.legend()
            plt.title(f"Baseline Test Case {i} column {j}")
            plt.savefig(f"{multi_output_path}/test_case_{i}/column_{j}.png")

print("[TimeXer] Masked MSE on test set:", mse_total_multi / count_multi)
with open("test_results.txt", "a") as log_file:
    print("Writing to file...")
    log_file.write(output_path + "\n")
    log_file.write("[TimeXer] Masked MSE on test set:" + str(mse_total_multi / count_multi) + "\n")



