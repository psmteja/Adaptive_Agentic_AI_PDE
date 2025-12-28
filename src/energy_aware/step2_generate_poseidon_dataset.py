"""
Step 2 (hardcoded): Use Poseidon to generate a synthetic (input, time, target) dataset.

Hardcoded settings:
    model_size  = "T"
    num_samples = 32
    batch_size  = 8
    outfile     = "data/poseidon_synth_T_32.pt"

Run:
    python step2_generate_poseidon_dataset_fixed.py
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from scOT.model import ScOT


# -----------------------------
# Poseidon backend (inline)
# -----------------------------

@dataclass
class PoseidonBackendConfig:
    model_size: Literal["T", "B", "L"] = "T"
    device: str = "auto"          # "auto", "cpu", or "cuda"
    dtype: str = "float32"        # or "float16"


class PoseidonBackend:
    def __init__(self, cfg: PoseidonBackendConfig):
        self.cfg = cfg

        # Resolve device
        if cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(cfg.device)

        # Resolve dtype
        if cfg.dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        repo_id = f"camlab-ethz/Poseidon-{cfg.model_size}"
        print(f"[PoseidonBackend] Loading model: {repo_id} on {self.device}...")

        self.model: ScOT = ScOT.from_pretrained(repo_id)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        config = self.model.config
        print("[PoseidonBackend] Loaded config:")
        print(f"  image_size      = {config.image_size}")
        print(f"  num_channels    = {config.num_channels}")
        print(f"  num_out_channels= {config.num_out_channels}")
        print(f"  depths          = {config.depths}")
        print(f"  num_heads       = {config.num_heads}")
        print(f"  residual_model  = {config.residual_model}")
        print(f"  use_conditioning= {config.use_conditioning}")
        print()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[PoseidonBackend] Total parameters: {n_params:,}")
        print()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Thin wrapper around Poseidon forward pass.

        Args:
            x: (B, C_in, H, W)
            t: (B,) time parameter

        Returns:
            y: (B, C_out, H, W)
        """
        start = time.time()
        out = self.model(pixel_values=x, time=t, return_dict=True)
        y = out.output
        elapsed = time.time() - start
        print(f"[PoseidonBackend] Forward: batch={x.size(0)} "
              f"took {elapsed:.4f} s")
        return y


# -----------------------------
# Hardcoded dataset generation
# -----------------------------

def main():
    # Hardcoded settings (this is your CLI command baked into code)
    model_size = "T"
    num_samples = 32
    batch_size = 8
    outfile = "data/poseidon_synth_T_32.pt"

    # You can change these if needed
    device = "auto"       # "auto", "cpu", or "cuda"
    dtype_str = "float32" # "float32" or "float16"

    print("[Step2] Hardcoded config:")
    print(f"  model_size  = {model_size}")
    print(f"  num_samples = {num_samples}")
    print(f"  batch_size  = {batch_size}")
    print(f"  outfile     = {outfile}")
    print(f"  device      = {device}")
    print(f"  dtype       = {dtype_str}")
    print()

    cfg = PoseidonBackendConfig(
        model_size=model_size,
        device=device,
        dtype=dtype_str,
    )
    backend = PoseidonBackend(cfg)

    # Get config info from Poseidon model
    mcfg = backend.model.config
    H = W = mcfg.image_size
    C_in = mcfg.num_channels
    C_out = mcfg.num_out_channels

    total = num_samples
    bs = batch_size

    # Decide dtype
    if dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device_torch = backend.device

    print("[Step2] Preparing tensors to hold the dataset...")
    inputs = torch.empty(total, C_in, H, W, dtype=dtype)
    times = torch.empty(total, dtype=dtype)
    targets = torch.empty(total, C_out, H, W, dtype=dtype)

    # Generate in batches
    idx = 0
    while idx < total:
        curr_bs = min(bs, total - idx)
        print(f"[Step2] Generating batch: idx={idx} .. {idx+curr_bs-1}")

        # Random input fields (for now; later this can be real PDE inputs)
        x = torch.randn(curr_bs, C_in, H, W, device=device_torch, dtype=dtype)

        # Random time parameter in [0, 1]
        t = torch.rand(curr_bs, device=device_torch, dtype=dtype)

        # Poseidon forward
        with torch.no_grad():
            y = backend.forward(x, t)  # (curr_bs, C_out, H, W)

        # Move to CPU buffers for saving
        inputs[idx:idx+curr_bs] = x.cpu()
        times[idx:idx+curr_bs] = t.cpu()
        targets[idx:idx+curr_bs] = y.cpu()

        idx += curr_bs

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = {
        "inputs": inputs,   # shape: (N, C_in, H, W)
        "times": times,     # shape: (N,)
        "targets": targets, # shape: (N, C_out, H, W)
        "meta": {
            "model_size": model_size,
            "image_size": H,
            "num_in_channels": C_in,
            "num_out_channels": C_out,
            "dtype": str(dtype),
        },
    }

    torch.save(data_dict, out_path)
    print(f"[Step2] Saved dataset with {total} samples to: {out_path.resolve()}")
    print(f"[Step2] Input shape   : {tuple(inputs.shape)}")
    print(f"[Step2] Target shape  : {tuple(targets.shape)}")
    print(f"[Step2] Time shape    : {tuple(times.shape)}")
    print("[Step2] Done ✅")


if __name__ == "__main__":
    main()
