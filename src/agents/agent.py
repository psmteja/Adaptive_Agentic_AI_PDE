from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import torch

from pde_data import generate_heat_data
from pde_tools import (
    load_pde_transformer_model,
    rollout_pde_transformer,
    compute_metrics,
    plot_rollout,
)


@dataclass
class ExperimentConfig:
    experiment_name: str
    nx: int
    ny: int
    nt: int
    dt: float
    nu: float
    model_variant: str
    max_horizon: int
    horizons_to_eval: List[int]
    output_dir: str
    horizon_for_plot: int


class ExperimentAgent:
    """Simple non-LLM experiment runner.

    In an actual "agent AI" setup, the decision-making logic here would be
    driven by an LLM that chooses which tools to call. This class is a
    lightweight stand-in that orchestrates the calls in a fixed sequence.
    """

    def __init__(self, config: ExperimentConfig, device: str | None = None):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[Agent] Using device: {self.device}")
        print(f"[Agent] Loading PDE-Transformer variant: {self.config.model_variant}")
        self.model = load_pde_transformer_model(
            device=self.device,
            variant=self.config.model_variant,
        )

    def run(self) -> Dict[int, Dict[str, float]]:
        """Run the experiment and return metrics for each horizon."""
        cfg = self.config

        print("[Agent] Generating baseline heat-equation data...")
        u = generate_heat_data(
            nx=cfg.nx,
            ny=cfg.ny,
            nt=cfg.nt,
            dt=cfg.dt,
            nu=cfg.nu,
        )
        print(f"[Agent] Data shape: {u.shape} (time, x, y)")

        # Use t0, t1 as initial states for the neural rollout
        u_t0, u_t1 = u[0], u[1]

        print(f"[Agent] Rolling out PDE-Transformer for {cfg.max_horizon} steps...")
        pred_seq = rollout_pde_transformer(
            self.model,
            u_t0=u_t0,
            u_t1=u_t1,
            n_steps=cfg.max_horizon,
            device=self.device,
        )

        metrics_by_horizon: Dict[int, Dict[str, float]] = {}

        for h in cfg.horizons_to_eval:
            idx = 1 + h  # prediction index in pred_seq and u
            if idx >= len(pred_seq) or idx >= u.shape[0]:
                print(f"[Agent] Skipping horizon {h}: insufficient time steps.")
                continue

            pred = pred_seq[idx]
            gt = u[idx]

            metrics = compute_metrics(pred, gt)
            metrics_by_horizon[h] = metrics

            print(f"[Agent] Horizon {h}: {metrics}")

        # Plot at a chosen horizon
        horizon_to_plot = cfg.horizon_for_plot
        print(f"[Agent] Saving plot for horizon {horizon_to_plot}...")
        plot_path = f"{cfg.output_dir}/rollout_h{horizon_to_plot}.png"
        plot_rollout(pred_seq, u, horizon=horizon_to_plot, output_path=plot_path)
        print(f"[Agent] Plot saved to: {plot_path}")

        return metrics_by_horizon
