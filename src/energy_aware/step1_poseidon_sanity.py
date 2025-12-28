"""
Step 1: Basic Poseidon setup + sanity check.

- Loads a pretrained Poseidon model (T/B/L).
- Creates a dummy PDE field input with the right shape.
- Runs a forward pass and prints shapes, param count, and timing.

Run:
    python step1_poseidon_sanity.py --model_size T

Model sizes (from Hugging Face collection):
    T  ~ tiny
    B  ~ base
    L  ~ large
"""

import argparse
import time
from dataclasses import dataclass
from typing import Literal

import torch

# Poseidon model class
from scOT.model import ScOT


@dataclass
class PoseidonBackendConfig:
    model_size: Literal["T", "B", "L"] = "T"
    device: str = "auto"          # "auto", "cpu", or "cuda"
    dtype: str = "float32"        # or "float16" if you want mixed precision later


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

        # Hugging Face repo id for Poseidon
        repo_id = f"camlab-ethz/Poseidon-{cfg.model_size}"
        print(f"[PoseidonBackend] Loading model: {repo_id} on {self.device}...")

        # Load pretrained Poseidon model
        self.model: ScOT = ScOT.from_pretrained(repo_id)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # Print a small config summary
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

        # Param count
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[PoseidonBackend] Total parameters: {n_params:,}")
        print()

    @torch.no_grad()
    def run_eval_forward(self, batch_size: int = 2):
        """
        Run a eval forward pass with random input to check that everything works.

        Poseidon expects:
          - pixel_values: (B, C_in, H, W)
          - time:         (B,)  continuous time parameter (used/ignored depending on config)
        """
        cfg = self.model.config

        # Use the model's configured image size and channel counts
        H = W = cfg.image_size
        C_in = cfg.num_channels
        C_out = cfg.num_out_channels

        print(f"[PoseidonBackend] Sending eval input:")
        print(f"  batch_size = {batch_size}")
        print(f"  input shape= (B={batch_size}, C_in={C_in}, H={H}, W={W})")

        # Random eval input field and time
        x = torch.randn(batch_size, C_in, H, W, device=self.device, dtype=self.dtype)
        # Time: one scalar per sample; can be 0 for now
        t = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        # Forward pass
        start = time.time()
        output = self.model(pixel_values=x, time=t, return_dict=True)
        elapsed = time.time() - start

        y = output.output  # (B, C_out, H, W)
        print(f"[PoseidonBackend] Forward pass done in {elapsed:.4f} seconds.")
        print(f"  output shape= {tuple(y.shape)} "
              f"(expected ~ (B, C_out={C_out}, H={H}, W={W}))")

        # Simple sanity check
        assert y.shape[0] == batch_size, "Batch size mismatch in output"
        assert y.shape[1] == C_out, "Output channel mismatch"
        print("[PoseidonBackend] Sanity checks passed ✅")

        return y


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Poseidon sanity check")
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["T", "B", "L"],
        default="T",
        help="Poseidon model size to load (T, B, or L).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Tensor dtype for the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="eval batch size for the test forward pass.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PoseidonBackendConfig(
        model_size=args.model_size,
        device=args.device,
        dtype=args.dtype,
    )
    backend = PoseidonBackend(cfg)
    backend.run_eval_forward(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
