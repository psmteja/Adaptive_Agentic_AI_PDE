"""
Step 6: Build LLM training data from FNO experiment logs.

Reads:
    results/poseidon_fno_experiments.jsonl

Each line: JSON object produced by step4_fno_experiment_logger.py

We:
  - Load all experiments for a "task" (here: Poseidon-T 128x128 PDE surrogate)
  - Compute a combined energy/accuracy score (same idea as Step 5)
  - Build an instruction-style dataset:
        {"prompt": "...", "response": "..."}

Writes:
    llm_data/fno_planner_train.jsonl
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


# -----------------------------
# Config
# -----------------------------

RESULTS_FILE = "results/poseidon_fno_experiments.jsonl"
OUT_LLM_DATA = "llm_data/fno_planner_train.jsonl"

# Weights for energy-aware score
W_LOSS = 0.3   # accuracy
W_PARAMS = 0.35  # model size
W_TIME = 0.35    # speed / energy proxy


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

    print(f"[Step6] Loaded {len(records)} experiments from {path.resolve()}")
    return records


def normalize(values: List[float]) -> List[float]:
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def build_single_example(experiments: List[ExperimentRecord]) -> Dict[str, str]:
    """
    Build one prompt/response pair from all experiments.
    Later you can extend this to multiple tasks.
    """
    if not experiments:
        raise ValueError("No experiments to build example from.")

    # Collect raw metrics
    losses = [e.final_eval_loss for e in experiments]
    params = [float(e.num_params) for e in experiments]
    times = [e.avg_epoch_time_s for e in experiments]

    # Normalize
    loss_norm = normalize(losses)
    params_norm = normalize(params)
    time_norm = normalize(times)

    # Compute scores
    scores = []
    for ln, pn, tn in zip(loss_norm, params_norm, time_norm):
        score = W_LOSS * ln + W_PARAMS * pn + W_TIME * tn
        scores.append(score)

    best_idx = min(range(len(experiments)), key=lambda i: scores[i])
    best = experiments[best_idx]

    # ---- Build prompt text ----
    # Describe the "task"
    meta = best.dataset_meta
    task_desc = (
        f"PDE surrogate task using Poseidon teacher; "
        f"model_size={meta.get('model_size', 'unknown')}, "
        f"image_size={meta.get('image_size', 'unknown')}, "
        f"num_in_channels={meta.get('num_in_channels', 'unknown')}, "
        f"num_out_channels={meta.get('num_out_channels', 'unknown')}."
    )

    # Describe energy objective
    weights_desc = (
        f"We want an energy-efficient FNO configuration that balances accuracy, "
        f"model size, and speed. We use a weighted score:\n"
        f"  score = {W_LOSS} * loss_norm + {W_PARAMS} * params_norm + {W_TIME} * time_norm\n"
        f"where lower is better, and each 'norm' is min-max normalized across the candidates."
    )

    # List candidate configurations
    lines = []
    for i, e in enumerate(experiments):
        lines.append(
            f"- {e.label}: width={e.width}, num_layers={e.num_layers}, "
            f"modes_x={e.modes_x}, modes_y={e.modes_y}, "
            f"num_params={e.num_params}, "
            f"final_eval_loss={e.final_eval_loss:.6f}, "
            f"avg_epoch_time_s={e.avg_epoch_time_s:.3f}"
        )

    candidates_desc = "\n".join(lines)

    prompt = (
        "You are an AI assistant that designs energy-efficient Fourier Neural Operators (FNOs) "
        "for PDE surrogate modeling tasks.\n\n"
        f"Task description:\n{task_desc}\n\n"
        f"Objective:\n{weights_desc}\n\n"
        "We ran several FNO configurations and measured their performance:\n"
        f"{candidates_desc}\n\n"
        "Question:\n"
        "1. Which configuration is the best under this objective?\n"
        "2. Justify your choice by explaining the trade-offs between accuracy, model size, and speed.\n"
        "3. Then restate the recommended configuration in a clear, machine-readable JSON block with keys:\n"
        "   {\"label\", \"width\", \"num_layers\", \"modes_x\", \"modes_y\", \"num_params\"}.\n"
    )

    # ---- Build response text (what we want the LLM to learn to do) ----

    # Raw ranges for explanation
    loss_min, loss_max = min(losses), max(losses)
    param_min, param_max = min(params), max(params)
    time_min, time_max = min(times), max(times)

    response = []
    response.append(f"The best configuration under the given weighted objective is '{best.label}'.")
    response.append("")
    response.append("Reasoning:")
    response.append(
        f"- Accuracy: It has eval loss {best.final_eval_loss:.6f}, "
        f"within the range [{loss_min:.6f}, {loss_max:.6f}] seen in the experiments, "
        f"and it is relatively strong compared to the others."
    )
    response.append(
        f"- Model size: It uses {best.num_params:,} parameters, "
        f"compared to the range [{param_min:.0f}, {param_max:.0f}]."
    )
    response.append(
        f"- Speed: It runs in {best.avg_epoch_time_s:.3f} seconds per epoch "
        f"on {best.device}, compared to the range [{time_min:.3f}, {time_max:.3f}]."
    )
    response.append(
        "Given the chosen weights, this configuration provides a good balance between "
        "low eval loss and acceptable size/speed. More aggressive compression would "
        "reduce parameters and speed further but at the cost of noticeably worse accuracy."
    )
    response.append("")
    response.append("Recommended configuration in JSON:")
    response.append("```json")
    response.append(json.dumps({
        "label": best.label,
        "width": best.width,
        "num_layers": best.num_layers,
        "modes_x": best.modes_x,
        "modes_y": best.modes_y,
        "num_params": best.num_params,
    }, indent=2))
    response.append("```")

    response_text = "\n".join(response)

    return {
        "prompt": prompt,
        "response": response_text,
    }


def main():
    experiments = load_experiments(RESULTS_FILE)
    if not experiments:
        print("[Step6] No experiments found, nothing to build.")
        return

    out_path = Path(OUT_LLM_DATA)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    example = build_single_example(experiments)

    # For now: write a single training example.
    # Later, as you get more tasks/datasets, you can append more examples.
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")

    print(f"[Step6] Wrote 1 LLM training example to: {out_path.resolve()}")
    print("[Step6] Example prompt snippet:")
    print("----------------------------------------")
    print(example["prompt"][:400] + " ...")
    print("----------------------------------------")
    print("You can open the JSONL file to see the full prompt and response.")


if __name__ == "__main__":
    main()
