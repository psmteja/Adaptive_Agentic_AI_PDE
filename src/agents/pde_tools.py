from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from pdetransformer.core.mixed_channels import PDETransformer


# -------------------------- Model utilities --------------------------

def load_pde_transformer_model(device: str = "cpu",
                               variant: str = "mc-s") -> PDETransformer:
    """Load a PDE-Transformer model from Hugging Face.

    Args:
        device: 'cpu' or 'cuda'
        variant: one of ['mc-s', 'mc-b', 'mc-l', 'sc-s', 'sc-b', 'sc-l']

    Returns:
        An instance of PDETransformer in eval mode on the requested device.
    """
    model = PDETransformer.from_pretrained(
        "thuerey-group/pde-transformer",
        subfolder=variant,
    ).to(device)
    model.eval()
    return model


def _unwrap_pde_output(y_out: Any) -> torch.Tensor:
    """Convert PDE-Transformer's output to a plain torch.Tensor.

    Newer versions of pdetransformer return a PDEOutput object that
    contains a .prediction tensor. Older versions may already return a
    raw tensor. This helper tries to handle both.
    """
    if isinstance(y_out, torch.Tensor):
        return y_out

    # PDEOutput in current versions exposes .prediction
    if hasattr(y_out, "prediction"):
        return y_out.prediction

    # Fallback: if it's dict-like, try first value
    try:
        values = list(y_out.values())  # type: ignore[attr-defined]
        if len(values) > 0 and isinstance(values[0], torch.Tensor):
            return values[0]
    except Exception:
        pass

    raise TypeError(f"Cannot unwrap PDE output of type {type(y_out)}")


def run_pde_transformer_step(model: PDETransformer,
                             u_t0: np.ndarray,
                             u_t1: np.ndarray,
                             device: str = "cpu") -> np.ndarray:
    """Return PDE-Transformer's prediction for the next time step.

    Args:
        model: PDE-Transformer model (already on the right device)
        u_t0: array of shape (nx, ny) for time t0
        u_t1: array of shape (nx, ny) for time t1
        device: 'cpu' or 'cuda'

    Returns:
        Predicted next state as a NumPy array of shape (nx, ny).
    """
    assert u_t0.shape == u_t1.shape, "u_t0 and u_t1 must have same shape"

    nx, ny = u_t0.shape
    x_in = np.stack([u_t0, u_t1], axis=0)  # (2, nx, ny)
    x_in = torch.from_numpy(x_in)[None, ...]  # (1, 2, nx, ny)
    x_in = x_in.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        y_out = model(x_in)
    y = _unwrap_pde_output(y_out)  # (1, 2, nx, ny)

    # Interpret channel 1 as "next state" prediction
    y_next = y[0, 1].detach().cpu().numpy()
    return y_next


def rollout_pde_transformer(model: PDETransformer,
                            u_t0: np.ndarray,
                            u_t1: np.ndarray,
                            n_steps: int,
                            device: str = "cpu") -> List[np.ndarray]:
    """Roll forward n_steps using PDE-Transformer.

    Args:
        model: PDE-Transformer model
        u_t0: state at time t0, shape (nx, ny)
        u_t1: state at time t1, shape (nx, ny)
        n_steps: number of steps to predict
        device: 'cpu' or 'cuda'

    Returns:
        List of states [u_t0, u_t1, u_t2_pred, ..., u_t{n_steps+1}_pred],
        each an array of shape (nx, ny).
    """
    states: List[np.ndarray] = [u_t0, u_t1]
    curr_prev = u_t0
    curr = u_t1

    for _ in range(n_steps):
        u_next = run_pde_transformer_step(model, curr_prev, curr, device=device)
        states.append(u_next)
        curr_prev, curr = curr, u_next

    return states


# ---------------------------- Metrics ----------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute simple error metrics between prediction and ground truth."""
    assert pred.shape == gt.shape, "pred and gt must have same shape"

    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    # Relative L2 error (against gt)
    gt_norm = float(np.sqrt(np.sum(gt ** 2))) + 1e-12
    rel_l2 = float(np.sqrt(np.sum(diff ** 2)) / gt_norm)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "rel_l2": rel_l2,
    }


# ---------------------------- Plotting ----------------------------

def plot_rollout(pred_seq: List[np.ndarray],
                 gt_seq: np.ndarray,
                 horizon: int,
                 output_path: str) -> None:
    """Save a comparison plot of ground truth vs prediction.

    Args:
        pred_seq: list of predictions [u0, u1, ..., u_{n_steps+1}]
        gt_seq: baseline solution array of shape (nt, nx, ny)
        horizon: prediction horizon to show (number of steps ahead from t1)
        output_path: filename to save the plot to
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # We interpret:
    #   - pred_seq[1 + horizon]  as prediction at t_{1 + horizon}
    #   - gt_seq[1 + horizon]    as ground truth at t_{1 + horizon}
    idx = 1 + horizon
    pred = pred_seq[idx]
    gt = gt_seq[idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(gt_seq[0], origin="lower")
    axes[0].set_title("u(t0)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gt, origin="lower")
    axes[1].set_title(f"Ground truth u(t{idx})")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(pred, origin="lower")
    axes[2].set_title(f"PDE-Transformer pred u(t{idx})")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
