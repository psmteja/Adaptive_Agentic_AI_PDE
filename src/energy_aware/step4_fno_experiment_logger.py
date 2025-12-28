"""
Step 4: FNO experiment logger on Poseidon synthetic dataset.

- Loads data/poseidon_synth_T_32.pt
- Runs several FNO configurations
- Logs metrics (loss, params, timing) to a JSONL file

Run:
    python step4_fno_experiment_logger.py

Sai Notes:
Trains it for a few epochs.
Measures how well it fits the data (loss).
Measures how long each epoch takes.
Counts how many parameters the model has.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================
# Dataset (same as step 3)
# ==========================

class PoseidonSyntheticDataset(Dataset):
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

    def __getitem__(self, idx: int):
        x = self.inputs[idx]    # (C_in, H, W)
        y = self.targets[idx]   # (C_out, H, W)
        t = self.times[idx]     # scalar
        return x, t, y


# ==========================
# FNO building blocks (same as step 3)
# ==========================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x_ft: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W_ft = x_ft.shape
        C_out = self.out_channels
        out_ft = torch.zeros(B, C_out, H, W_ft, dtype=torch.cfloat, device=x_ft.device)

        mx = min(self.modes_x, H)
        my = min(self.modes_y, W_ft)

        out_ft[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :my],
            self.weights[:, :, :mx, :my],
        )
        return out_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = self.compl_mul2d(x_ft)
        y = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")
        return y


class FNO2d(nn.Module):
    """
    FNO block with time as extra channel.
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

        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

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
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        t_channel = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x_cat = torch.cat([x, t_channel], dim=1)

        x = self.fc0(x_cat)
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            y_spec = spec(x)
            y_pw = pw(x)
            x = x + self.activation(y_spec + y_pw)

        x = self.activation(self.fc1(x))
        y = self.fc2(x)
        return y


# ==========================
# Experiment config & utils
# ==========================

@dataclass
class FNOExperimentConfig:
    width: int
    num_layers: int
    modes_x: int
    modes_y: int
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-3
    label: str = ""       # optional name


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


def run_experiment(
    cfg: FNOExperimentConfig,
    dataset: PoseidonSyntheticDataset,
    device: torch.device,
) -> dict:
    print(f"\n[Experiment] Starting config: {cfg}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    C_in = dataset.inputs.shape[1]
    C_out = dataset.targets.shape[1]

    model = FNO2d(
        in_channels=C_in + 1,
        out_channels=C_out,
        width=cfg.width,
        num_layers=cfg.num_layers,
        modes_x=cfg.modes_x,
        modes_y=cfg.modes_y,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Experiment] Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    epoch_times: List[float] = []
    train_losses: List[float] = []
    eval_losses: List[float] = []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        eval_loss = evaluate(model, loader, loss_fn, device)
        dt = time.time() - t0

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        epoch_times.append(dt)

        print(f"  Epoch {epoch:02d} | train_loss = {train_loss:.6f} "
              f"| eval_loss = {eval_loss:.6f} | time = {dt:.2f} s")

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "num_params": n_params,
        "final_train_loss": train_losses[-1],
        "final_eval_loss": eval_losses[-1],
        "avg_epoch_time_s": sum(epoch_times) / len(epoch_times),
        "device": str(device),
        "dataset_size": len(dataset),
        "dataset_meta": dataset.meta,
    }
    print(f"[Experiment] Done. final_eval_loss={result['final_eval_loss']:.6f}, "
          f"avg_epoch_time={result['avg_epoch_time_s']:.2f}s")

    return result


# ==========================
# Main
# ==========================

def main():
    data_file = "data/poseidon_synth_T_32.pt"
    results_file = "results/poseidon_fno_experiments.jsonl"

    print("[Step4] Loading dataset...")
    dataset = PoseidonSyntheticDataset(data_file)
    device = get_device()
    print(f"[Step4] Using device: {device}")

    # Define a few experiment configs (you can extend this list)
    experiments = [
        FNOExperimentConfig(
            width=32, num_layers=4, modes_x=16, modes_y=16,
            num_epochs=5, label="baseline_32w_4L_16m"
        ),
        FNOExperimentConfig(
            width=32, num_layers=4, modes_x=8, modes_y=8,
            num_epochs=5, label="fewer_modes_32w_4L_8m"
        ),
        FNOExperimentConfig(
            width=16, num_layers=3, modes_x=8, modes_y=8,
            num_epochs=5, label="smaller_model_16w_3L_8m"
        ),
    ]

    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a", encoding="utf-8") as f:
        for cfg in experiments:
            result = run_experiment(cfg, dataset, device)
            f.write(json.dumps(result) + "\n")

    print(f"\n[Step4] Appended {len(experiments)} experiment results to: "
          f"{results_path.resolve()}")
    print("[Step4] Done ✅")


if __name__ == "__main__":
    main()
