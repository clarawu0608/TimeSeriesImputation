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
from scipy.signal import find_peaks

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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Output path
output_path = "outputs/frequency/test4_{}_{}_{}_{}_{}_{}/".format(
    args.seq_len, args.token_size, args.token_overlap, args.missing_type, args.lm, args.missing_rate)
os.makedirs(output_path, exist_ok=True)

# Read data
df = pd.read_csv("dataset/periodic_signal.csv", parse_dates=["date"])
data = df.iloc[:, 1].values.astype(np.float32)

df = df.sort_values("date")  # just in case
timestamps = df["date"].values
dt_seconds = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 's')  # in seconds
sampling_rate = 1 / dt_seconds  # in Hz (cycles per second)

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
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
train_indices = train_set.indices
val_indices = val_set.indices
test_indices = test_set.indices

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

    def forward(self, x, mask=None):
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

model = TransformerEncoder().to(device)
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
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(masked, mask)
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step(), mask
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = model(masked, mask)
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

# For denormalization
def denormalize(x, data_std, data_mean):
    return x * data_std + data_mean

# For reconstructing overlapping predictions (assuming summing and counting)
def reconstruct(pred_patches, count_map):
    return np.divide(pred_patches, count_map, out=np.zeros_like(pred_patches), where=count_map != 0)


model.load_state_dict(torch.load("best_model.pth"))
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
    mse_total += ((out_denorm[~mask_np] - gt_denorm[~mask_np]) ** 2).sum()
    count += (~mask_np).sum()

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


# === Full Data Imputation and FFT ===
print("Imputing full dataset...")

# Prepare masked_data tensor
data_tensor = torch.tensor(masked_data).unsqueeze(0).to(torch.float32).to(device)  # [1, L]
mask_tensor = torch.tensor(mask_matrix).unsqueeze(0).to(torch.bool).to(device)  # [1, L] 

with torch.no_grad():
    model.eval()
    imputated = model(data_tensor, mask_tensor).squeeze(0).cpu().numpy()  # [L]

# Fill in the missing values in the original masked data
imputed_series = masked_data.copy()
imputed_series[~mask_matrix] = imputated[~mask_matrix]

# Save the imputed series
np.save(f"{output_path}imputed_series.npy", imputed_series)
plt.figure()
plt.plot(imputed_series, label="Imputed")
plt.plot(data, label="Original", alpha=0.5)
plt.legend()
plt.title("Imputed vs Original Series")
plt.savefig(f"{output_path}imputed_vs_original.png")

# FFT
print("Performing FFT on the imputed series...")
fft_vals = np.fft.fft(imputed_series)
fft_freqs = np.fft.fftfreq(len(imputed_series), d=1/sampling_rate)

scale_factor = 1e5
bandwidth = 0.0000005 * scale_factor 
fft_freqs = fft_freqs * scale_factor

# Plot magnitude spectrum
plt.figure()
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_vals[:len(fft_vals)//2]))
plt.title("FFT Magnitude Spectrum of Imputed Series")
plt.xlabel(f"Frequency ({1/scale_factor} Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_path}fft_spectrum_impute1.png")
print("FFT completed and saved.")

# Find dominate frequency
fft_magnitudes = np.abs(fft_vals)

# Keep only positive frequencies
positive_freqs = fft_freqs[:len(fft_freqs)//2]
positive_mags = fft_magnitudes[:len(fft_magnitudes)//2]

# Threshold-based selection (e.g., top 5 peaks above a percentile)
def get_dominant_frequencies(magnitudes, freqs, multiplier=18.0, use='median'):
    """
    Identify dominant frequency peaks using local maxima and a dynamic threshold.
    
    Parameters:
        magnitudes (np.ndarray): FFT magnitude spectrum.
        freqs (np.ndarray): Corresponding frequency values.
        multiplier (float): Multiplier for thresholding based on median/mean.
        use (str): 'median' or 'mean' to define the baseline noise level.
        
    Returns:
        dominant_freqs (np.ndarray): Frequencies of dominant peaks.
        dominant_mags (np.ndarray): Magnitudes of dominant peaks.
        dominant_indices (np.ndarray): Indices of dominant peaks.
    """
    baseline = np.median(magnitudes) if use == 'median' else np.mean(magnitudes)
    threshold = multiplier * baseline
    
    # Find local maxima that are higher than the threshold
    peak_indices, properties = find_peaks(magnitudes, height=threshold)
    peak_mags = properties['peak_heights']
    
    dominant_freqs = freqs[peak_indices]
    dominant_mags = magnitudes[peak_indices]

    return dominant_freqs, dominant_mags, peak_indices

dominant_freqs, dominant_mags, dominant_indices = get_dominant_frequencies(
    magnitudes=positive_mags, 
    freqs=positive_freqs, 
    multiplier=18.0, 
    use='median'
)

print(f"Identified {len(dominant_freqs)} dominant frequencies ({1/scale_factor}Hz): {dominant_freqs}")

def merge_close_frequencies(freqs, min_gap):
    """
    Merge frequencies that are closer than min_gap.
    
    Parameters:
        freqs (np.ndarray): Array of dominant frequencies (sorted).
        min_gap (float): Minimum allowed distance between frequencies.
        
    Returns:
        merged_freqs (list): Filtered dominant frequencies with no close neighbors.
    """
    if len(freqs) == 0:
        return []

    freqs = np.sort(freqs)
    merged_freqs = [freqs[0]]

    for f in freqs[1:]:
        if np.abs(f - merged_freqs[-1]) >= min_gap:
            merged_freqs.append(f)

    return merged_freqs

filtered_dominant_freqs = merge_close_frequencies(dominant_freqs, min_gap=2*bandwidth)
print(f"Filtered {len(filtered_dominant_freqs)} dominant frequencies ({1/scale_factor}Hz): {filtered_dominant_freqs}")

# Band pass filtering
def extract_component(series, freq, sampling_rate, bandwidth, scale_factor):
    fft_vals = np.fft.fft(series)
    fft_freqs = np.fft.fftfreq(len(series), d=1/sampling_rate)
    fft_freqs = fft_freqs * scale_factor

    # Create a mask that isolates the frequency band
    mask = (np.abs(fft_freqs - freq) < bandwidth) | (np.abs(fft_freqs + freq) < bandwidth)
    filtered_fft = np.where(mask, fft_vals, 0)
    return np.fft.ifft(filtered_fft).real

imputed_series_residual = imputed_series.copy()
components = []
gt_components = []  # for training/validation targets
for freq in filtered_dominant_freqs:
    comp_from_imputed = extract_component(imputed_series, freq, sampling_rate=1/dt_seconds, bandwidth=bandwidth, scale_factor=scale_factor)
    imputed_series_residual -= comp_from_imputed
    comp_from_gt = extract_component(data, freq, sampling_rate=1/dt_seconds, bandwidth=bandwidth, scale_factor=scale_factor)
    components.append(comp_from_imputed)  # inputs to model
    gt_components.append(comp_from_gt)    # targets (true)


# Train and save each component
reconstructed_series = np.zeros(len(data))
reconstructed_mask = np.zeros(len(data), dtype=bool)
predicted_components = np.zeros(len(data))

for i, freq in enumerate(filtered_dominant_freqs):
    print(f"\n=== Component {i}: Frequency = {freq:.4f} (1/hour) ===")
    comp_output_path = f"{output_path}/component_{i}/"
    os.makedirs(comp_output_path, exist_ok=True)

    comp_input = components[i]
    comp_gt = gt_components[i]

    # Plot raw (unnormalized) component
    plt.figure(figsize=(10, 3))
    plt.plot(comp_input, label="Extracted Component")
    plt.plot(comp_gt, label="Ground Truth", alpha=0.5)
    plt.legend()
    plt.title(f"Raw Extracted Component {i} (Freq: {freq:.4e} Hz)")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comp_output_path}/raw.png")
    plt.close()

    # Normalize
    mean, std = np.mean(comp_input), np.std(comp_input)
    print(f"[Component {i}] Mean: {mean:.4f}, Std: {std:.4f}")
    if std < 1e-8:
        print(f"[Component {i}] Skipped due to near-zero std.")
        continue

    norm_comp_input = (comp_input - mean) / std
    norm_comp_gt = (comp_gt - mean) / std  # Normalize GT using input stats
    masked_comp = norm_comp_input.copy()
    masked_comp[~mask_matrix] = 0.0

    # Create dataset using same indices
    dataset = TimeSeriesDataset(norm_comp_gt, masked_comp, mask_matrix, args.seq_len)
    train_set_i = torch.utils.data.Subset(dataset, train_indices)
    val_set_i = torch.utils.data.Subset(dataset, val_indices)
    test_set_i = torch.utils.data.Subset(dataset, test_indices)
    train_loader = DataLoader(train_set_i, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set_i, batch_size=args.batch_size)
    test_loader = DataLoader(test_set_i, batch_size=1)

    # Model setup
    model = TransformerEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)

    # Logging
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []
    

    # === Training loop
    for epoch in range(args.epoch_num):
        model.train()
        train_loss = 0
        for gt, masked, mask in train_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(masked, mask)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for gt, masked, mask in val_loader:
                gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
                out = model(masked, mask)
                loss = criterion(out, gt)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"[Comp {i}] Epoch {epoch}, Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")
        scheduler.step()

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{comp_output_path}best_model.pth")
            last_attn_weights = model.attn_weights
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print(f"[Comp {i}] Early stopping.")
                break

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title(f"Loss Curve (Component {i})")
    plt.savefig(f"{comp_output_path}loss_curve.png")
    plt.close()

    if last_attn_weights is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(last_attn_weights[0].cpu(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Attention Map (Component {i})")
        plt.savefig(f"{comp_output_path}attention_map.png")
        plt.close()

    # === Test and Imputation
    print(f"[Comp {i}] Testing and imputating...")
    model.load_state_dict(torch.load(f"{comp_output_path}best_model.pth"))
    model.eval()

    predicted_comp = np.zeros_like(data)
    count_map = np.zeros_like(data)
    comp_mse_total = 0
    comp_count = 0
    sample_plot_count = 0

    with torch.no_grad():
        for j, (gt, masked, mask) in enumerate(test_loader):
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            idx = test_indices[j]
            out = model(masked, mask).squeeze(0).cpu().numpy()
            predicted_comp[idx:idx + args.seq_len] += out
            count_map[idx:idx + args.seq_len] += 1
            reconstructed_mask[idx:idx + args.seq_len] |= ~mask.squeeze(0).cpu().numpy()


            # Plot first two test samples
            if sample_plot_count < 2:
                plt.figure(figsize=(8, 3))
                plt.plot(out * std + mean, label="Imputed")
                plt.plot(gt.cpu().numpy() * std + mean, label="GT", alpha=0.5)
                plt.title(f"Component {i} - Test Sample {sample_plot_count}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{comp_output_path}test_sample_{sample_plot_count}.png")
                plt.close()
                sample_plot_count += 1

    # Average overlapping predictions
    predicted_comp = np.divide(predicted_comp, count_map, out=np.zeros_like(predicted_comp), where=count_map != 0)
    mean_pred, std_pred = np.mean(predicted_comp), np.std(predicted_comp)
    print(f"[Comp {i}] Mean: {mean_pred:.4f}, Std: {std_pred:.4f}")
    predicted_comp = predicted_comp * std + mean

    plt.figure(figsize=(8, 3))
    plt.plot(predicted_comp, label="Imputed")
    plt.title(f"Component {i} - predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comp_output_path}predicted_comp.png")
    plt.close()

    segment_len = 200  # adjust as needed
    segments = [500, 1000]  # starting indices of segments to show

    for k, start in enumerate(segments):
        end = start + segment_len
        plt.figure(figsize=(10, 3))
        plt.plot(np.arange(start, end), comp_gt[start:end], label="Ground Truth", alpha=0.5)
        plt.plot(np.arange(start, end), predicted_comp[start:end], label="Reconstructed", alpha=0.8)
        plt.title(f"Reconstruction Comparison (Segment {k+1})")
        plt.xlabel("Time Index")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{comp_output_path}reconstructed_segment_{k+1}.png")
        plt.close()

    predicted_components += predicted_comp

# === Final Evaluation
gt_signal = data
mask_flat = reconstructed_mask

# Reconstruct via residual correction
# component_sum = np.sum(predicted_components, axis=0)
reconstructed_series = predicted_components + imputed_series_residual
total_mse = np.mean((gt_signal[~mask_matrix] - reconstructed_series[~mask_matrix]) ** 2)


print(f"\n✅ Final Masked MSE after reconstruction: {total_mse:.6f}")
with open("test_results.txt", "a") as f:
    f.write(f"\nTotal reconstructed test MSE: {total_mse:.6f}\n")

# === Final Evaluation Plot (Zoomed Segments) ===
segment_len = 200  # adjust as needed
segments = [500, 1000]  # starting indices of segments to show

for k, start in enumerate(segments):
    end = start + segment_len
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(start, end), gt_signal[start:end], label="Ground Truth", alpha=0.5)
    plt.plot(np.arange(start, end), reconstructed_series[start:end], label="Reconstructed", alpha=0.8)
    
    mask_segment = reconstructed_mask[start:end]
    mask_vis = np.ma.masked_where(~mask_segment, mask_segment)
    plt.plot(np.arange(start, end), mask_vis * np.max(gt_signal), 'r.', label="Missing", markersize=2)

    plt.title(f"Reconstruction Comparison (Segment {k+1})")
    plt.xlabel("Time Index")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}reconstructed_segment_{k+1}.png")
    plt.close()

# FFT
print("Performing FFT on the imputed series...")
fft_vals = np.fft.fft(reconstructed_series)
fft_freqs = np.fft.fftfreq(len(reconstructed_series), d=1/sampling_rate)

scale_factor = 1e5
bandwidth = 0.0000005 * scale_factor 
fft_freqs = fft_freqs * scale_factor

# Plot magnitude spectrum
plt.figure()
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_vals[:len(fft_vals)//2]))
plt.title("FFT Magnitude Spectrum of Imputed Series")
plt.xlabel(f"Frequency ({1/scale_factor} Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_path}fft_spectrum_impute2.png")
print("FFT completed and saved.")