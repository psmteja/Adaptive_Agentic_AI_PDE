"""
Step 5: Simple energy-aware FNO recommender from experiment logs.

Reads:
    results/poseidon_fno_experiments.jsonl

Each line is a JSON object produced by step4_fno_experiment_logger.py, with keys:
    - config (width, num_layers, modes_x, modes_y, etc.)
    - num_params
    - final_eval_loss
    - avg_epoch_time_s
    - ...

We:
  - Load all experiments
  - Normalize loss, params, time
  - Compute a combined score:
        score = w_loss * loss_norm + w_params * params_norm + w_time * time_norm
  - Pick the best config and print a recommendation + explanation.

Run:
    python step5_recommend_fno_from_logs.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


# -----------------------------
# Config
# -----------------------------

RESULTS_FILE = "results/poseidon_fno_experiments.jsonl"

# Weights for the combined score
# You can tweak these later:
#   - w_loss: how much you care about accuracy
#   - w_params: how much you care about model size
#   - w_time: how much you care about speed / energy
W_LOSS = 0.5
W_PARAMS = 0.25
W_TIME = 0.25


@dataclass
class ExperimentRecord:
    label: str
    width: int
    num_layers: int
    modes_x: int
    modes_y: int
    batch_size: int
    num_epochs: int
    learning_rate: float

    num_params: int
    final_eval_loss: float
    avg_epoch_time_s: float
    device: str
    dataset_size: int
    dataset_meta: Dict[str, Any]


def load_experiments(results_file: str) -> List[ExperimentRecord]:
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path.resolve()}")

    records: List[ExperimentRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            cfg = obj["config"]
            rec = ExperimentRecord(
                label=cfg.get("label", ""),
                width=cfg["width"],
                num_layers=cfg["num_layers"],
                modes_x=cfg["modes_x"],
                modes_y=cfg["modes_y"],
                batch_size=cfg["batch_size"],
                num_epochs=cfg["num_epochs"],
                learning_rate=cfg["learning_rate"],
                num_params=obj["num_params"],
                final_eval_loss=obj["final_eval_loss"],
                avg_epoch_time_s=obj["avg_epoch_time_s"],
                device=obj.get("device", "unknown"),
                dataset_size=obj.get("dataset_size", -1),
                dataset_meta=obj.get("dataset_meta", {}),
            )
            records.append(rec)

    print(f"[Step5] Loaded {len(records)} experiments from {path.resolve()}")
    return records


def normalize(values: List[float]) -> List[float]:
    """
    Min-max normalize list of floats to [0, 1].
    If all values are equal, return zeros.
    """
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def main():
    experiments = load_experiments(RESULTS_FILE)
    if not experiments:
        print("[Step5] No experiments found, nothing to recommend.")
        return

    # Collect raw metrics
    losses = [e.final_eval_loss for e in experiments]
    params = [float(e.num_params) for e in experiments]
    times = [e.avg_epoch_time_s for e in experiments]

    # Normalize to [0, 1]
    loss_norm = normalize(losses)
    params_norm = normalize(params)
    time_norm = normalize(times)

    print("\n[Step5] Normalized metrics (0=best, 1=worst):")
    for e, ln, pn, tn in zip(experiments, loss_norm, params_norm, time_norm):
        print(f"  {e.label:25s} | loss_norm={ln:.3f} | params_norm={pn:.3f} | time_norm={tn:.3f}")

    # Compute combined scores
    scores = []
    for e, ln, pn, tn in zip(experiments, loss_norm, params_norm, time_norm):
        score = W_LOSS * ln + W_PARAMS * pn + W_TIME * tn
        scores.append(score)

    # Pick best experiment
    best_idx = min(range(len(experiments)), key=lambda i: scores[i])
    best = experiments[best_idx]

    print("\n[Step5] Combined scores (lower is better):")
    for e, s in zip(experiments, scores):
        print(f"  {e.label:25s} | score={s:.3f}")

    print("\n[Step5] Recommended configuration (given current weights):")
    print(f"  Label         : {best.label}")
    print(f"  width         : {best.width}")
    print(f"  num_layers    : {best.num_layers}")
    print(f"  modes_x/modes_y: {best.modes_x}/{best.modes_y}")
    print(f"  num_params    : {best.num_params:,}")
    print(f"  final_eval_loss: {best.final_eval_loss:.6f}")
    print(f"  avg_epoch_time: {best.avg_epoch_time_s:.3f} s/epoch")
    print(f"  device        : {best.device}")
    print(f"  dataset_size  : {best.dataset_size}")
    print(f"  dataset_meta  : {best.dataset_meta}")

    # Simple natural-language explanation
    print("\n[Step5] Explanation:")
    print(
        "  This configuration is chosen by balancing accuracy (eval loss), "
        "model size (parameter count), and speed (avg epoch time).\n"
        f"  - It has eval loss {best.final_eval_loss:.4f}, compared to the range "
        f"[{min(losses):.4f}, {max(losses):.4f}] seen in experiments.\n"
        f"  - It uses {best.num_params:,} parameters, compared to the range "
        f"[{min(params):.0f}, {max(params):.0f}].\n"
        f"  - It runs in {best.avg_epoch_time_s:.2f}s per epoch, compared to the range "
        f"[{min(times):.2f}, {max(times):.2f}].\n"
        "  You can adjust W_LOSS, W_PARAMS, and W_TIME at the top of this script "
        "if you want to prioritize accuracy vs energy differently."
    )

    print("\n[Step5] Done ✅")


if __name__ == "__main__":
    main()
