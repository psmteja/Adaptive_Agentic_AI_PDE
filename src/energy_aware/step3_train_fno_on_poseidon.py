"""
Step 3: Train a small FNO on the Poseidon synthetic dataset.

Assumes you already ran Step 2 and have:
    data/poseidon_synth_T_32.pt

This script:
  - Loads that .pt file
  - Builds a small FNO model (2D)
  - Trains it to map (inputs, time) -> targets
  - Saves the model checkpoint

Run:
    python step3_train_fno_on_poseidon.py
"""

import math
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================
# Dataset
# ==========================

class PoseidonSyntheticDataset(Dataset):
    """
    Wraps the torch .pt file from Step 2 as a PyTorch Dataset.

    Expects file with keys:
        - "inputs": (N, C_in, H, W)
        - "targets": (N, C_out, H, W)
        - "times": (N,)
    """
    def __init__(self, file_path: str):
        data = torch.load(file_path, map_location="cpu")
        self.inputs = data["inputs"].float()   # (N, C_in, H, W)
        self.targets = data["targets"].float() # (N, C_out, H, W)
        self.times = data["times"].float()     # (N,)
        self.meta = data.get("meta", {})

        assert self.inputs.shape[0] == self.targets.shape[0] == self.times.shape[0], \
            "Mismatch in number of samples"

        print("[Dataset] Loaded:")
        print(f"  inputs : {tuple(self.inputs.shape)}")
        print(f"  targets: {tuple(self.targets.shape)}")
        print(f"  times  : {tuple(self.times.shape)}")
        print(f"  meta   : {self.meta}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]    # (C_in, H, W)
        y = self.targets[idx]   # (C_out, H, W)
        t = self.times[idx]     # scalar
        return x, t, y


# ==========================
# FNO building blocks
# ==========================

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Does:
        - FFT
        - Truncate to low-frequency modes
        - Multiply by learned complex weights
        - iFFT

    Based on the standard FNO pattern, but kept small/simple.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Complex weights for the retained modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        x_ft: (B, C_in, H, W_ft)
        Returns:
        out_ft: (B, C_out, H, W_ft)
        """
        B, C_in, H, W_ft = x_ft.shape
        C_out = self.out_channels

        out_ft = torch.zeros(B, C_out, H, W_ft, dtype=torch.cfloat, device=x_ft.device)

        mx = min(self.modes_x, H)
        my = min(self.modes_y, W_ft)

        # Low-frequency block
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :my],
            self.weights[:, :, :mx, :my],
        )
        return out_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W) real-valued
        """
        B, C_in, H, W = x.shape
        # FFT: real-to-complex
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C_in, H, W_ft)
        # Multiply in Fourier space
        out_ft = self.compl_mul2d(x_ft)
        # Inverse FFT
        y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return y


class FNO2d(nn.Module):
    """
    Small 2D FNO for (input, time) -> target.

    - We treat `time` as an extra constant channel, broadcast over HxW.
    - Input channels: C_in (from dataset) + 1 (time).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 width: int = 32,
                 num_layers: int = 4,
                 modes_x: int = 16,
                 modes_y: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.num_layers = num_layers
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Lift input (C_in+1) to width with 1x1 conv
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        # A stack of spectral + pointwise layers
        self.spectral_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.spectral_layers.append(
                SpectralConv2d(width, width, modes_x, modes_y)
            )
            self.pointwise_layers.append(
                nn.Conv2d(width, width, kernel_size=1)
            )

        self.activation = nn.GELU()

        # Project back to output channels
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)    - input fields
        t: (B,)               - time scalar per sample

        Returns:
            y: (B, C_out, H, W)
        """
        B, C_in, H, W = x.shape

        # Add time as an extra channel, broadcast over HxW
        t_channel = t.view(B, 1, 1, 1).expand(-1, 1, H, W)  # (B, 1, H, W)
        x_cat = torch.cat([x, t_channel], dim=1)            # (B, C_in+1, H, W)

        # Lift to width
        x = self.fc0(x_cat)  # (B, width, H, W)

        # FNO blocks with residual connections
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            y_spec = spec(x)
            y_pw = pw(x)
            x = x + self.activation(y_spec + y_pw)

        # Project back to output channels
        x = self.activation(self.fc1(x))
        y = self.fc2(x)
        return y


# ==========================
# Training utilities
# ==========================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, t, y in loader:
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x, t)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, t, y in loader:
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)

        y_pred = model(x, t)
        loss = loss_fn(y_pred, y)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ==========================
# Main (hardcoded config)
# ==========================

def main():
    # -------- Hardcoded config (you can tweak here) --------
    data_file = "data/poseidon_synth_T_32.pt"
    out_checkpoint = "checkpoints/fno_poseidon_T.pt"

    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3

    width = 32
    num_layers = 4
    modes_x = 16
    modes_y = 16
    # -------------------------------------------------------

    print("[Step3] Config:")
    print(f"  data_file     = {data_file}")
    print(f"  out_checkpoint= {out_checkpoint}")
    print(f"  batch_size    = {batch_size}")
    print(f"  num_epochs    = {num_epochs}")
    print(f"  lr            = {learning_rate}")
    print(f"  width         = {width}")
    print(f"  num_layers    = {num_layers}")
    print(f"  modes_x/modes_y = {modes_x}/{modes_y}")
    print()

    # Dataset & loader
    dataset = PoseidonSyntheticDataset(data_file)
    # For now, just train & evaluate on the same tiny dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    C_in = dataset.inputs.shape[1]
    C_out = dataset.targets.shape[1]
    device = get_device()
    print(f"[Step3] Using device: {device}")

    model = FNO2d(
        in_channels=C_in + 1,   # C_in + time channel
        out_channels=C_out,
        width=width,
        num_layers=num_layers,
        modes_x=modes_x,
        modes_y=modes_y,
    ).to(device)

    print(f"[Step3] Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    print("[Step3] Starting training...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        eval_loss = evaluate(model, loader, loss_fn, device)
        dt = time.time() - t0

        print(f"  Epoch {epoch:02d} | train_loss = {train_loss:.6f} "
              f"| eval_loss = {eval_loss:.6f} | time = {dt:.2f} s")

    # Save checkpoint
    out_path = Path(out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "C_in": C_in,
                "C_out": C_out,
                "width": width,
                "num_layers": num_layers,
                "modes_x": modes_x,
                "modes_y": modes_y,
            },
        },
        out_path,
    )
    print(f"[Step3] Saved FNO checkpoint to: {out_path.resolve()}")
    print("[Step3] Done ✅")


if __name__ == "__main__":
    main()
