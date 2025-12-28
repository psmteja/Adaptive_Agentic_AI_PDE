"""
Step 7: Build multiple LLM training examples from FNO experiment logs.

We:
  - Load results/poseidon_fno_experiments.jsonl
  - Define several "objective profiles" (accuracy-first, balanced, energy-first)
  - For each profile, compute the best config under that objective
  - Emit a prompt/response pair describing the task, the candidates, the objective,
    and the chosen config with reasoning + JSON.

Output:
  llm_data/fno_planner_train_multi.jsonl
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


RESULTS_FILE = "results/poseidon_fno_experiments.jsonl"
OUT_LLM_DATA = "llm_data/fno_planner_train_multi.jsonl"


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


@dataclass
class ObjectiveProfile:
    name: str
    description: str
    w_loss: float
    w_params: float
    w_time: float
    # optional: accuracy tolerance ("within X% of best")
    max_loss_factor: float = 1.0  # 1.0 means no constraint; 1.05 = within 5% of best


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

    print(f"[Step7] Loaded {len(records)} experiments from {path.resolve()}")
    return records


def normalize(values: List[float]) -> List[float]:
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def build_example_for_objective(
    experiments: List[ExperimentRecord],
    obj: ObjectiveProfile,
) -> Dict[str, str]:
    if not experiments:
        raise ValueError("No experiments to build example from.")

    losses = [e.final_eval_loss for e in experiments]
    params = [float(e.num_params) for e in experiments]
    times = [e.avg_epoch_time_s for e in experiments]

    # Apply accuracy constraint: only keep configs within max_loss_factor of best
    best_loss = min(losses)
    max_allowed_loss = best_loss * obj.max_loss_factor
    candidate_indices = [
        i for i, e in enumerate(experiments) if e.final_eval_loss <= max_allowed_loss
    ]
    if not candidate_indices:
        # fallback: use all
        candidate_indices = list(range(len(experiments)))

    # Reduce metrics to candidates only
    cand_exps = [experiments[i] for i in candidate_indices]
    cand_losses = [losses[i] for i in candidate_indices]
    cand_params = [params[i] for i in candidate_indices]
    cand_times = [times[i] for i in candidate_indices]

    loss_norm = normalize(cand_losses)
    params_norm = normalize(cand_params)
    time_norm = normalize(cand_times)

    # Combined score for candidates
    scores = []
    for ln, pn, tn in zip(loss_norm, params_norm, time_norm):
        score = obj.w_loss * ln + obj.w_params * pn + obj.w_time * tn
        scores.append(score)

    best_idx_local = min(range(len(cand_exps)), key=lambda i: scores[i])
    best = cand_exps[best_idx_local]

    # Ranges for explanation (all experiments, not just candidates)
    loss_min, loss_max = min(losses), max(losses)
    param_min, param_max = min(params), max(params)
    time_min, time_max = min(times), max(times)

    # ---- Build prompt ----
    meta = best.dataset_meta
    task_desc = (
        f"PDE surrogate task using Poseidon teacher; "
        f"model_size={meta.get('model_size', 'unknown')}, "
        f"image_size={meta.get('image_size', 'unknown')}, "
        f"num_in_channels={meta.get('num_in_channels', 'unknown')}, "
        f"num_out_channels={meta.get('num_out_channels', 'unknown')}."
    )

    # List all candidates for transparency
    candidate_lines = []
    for e in experiments:
        candidate_lines.append(
            f"- {e.label}: width={e.width}, num_layers={e.num_layers}, "
            f"modes_x={e.modes_x}, modes_y={e.modes_y}, "
            f"num_params={e.num_params}, "
            f"final_eval_loss={e.final_eval_loss:.6f}, "
            f"avg_epoch_time_s={e.avg_epoch_time_s:.3f}"
        )
    candidates_desc = "\n".join(candidate_lines)

    weights_desc = (
        f"We are using the objective profile '{obj.name}':\n"
        f"{obj.description}\n\n"
        f"In numeric terms, we compute a weighted score on min-max normalized metrics:\n"
        f"  score = {obj.w_loss} * loss_norm + "
        f"{obj.w_params} * params_norm + "
        f"{obj.w_time} * time_norm\n"
        f"Lower score is better. Additionally, we only consider models whose eval loss "
        f"is <= {obj.max_loss_factor:.3f} × best_loss "
        f"(best_loss = {best_loss:.6f})."
    )

    prompt = (
        "You are an AI assistant that designs energy-efficient Fourier Neural Operators (FNOs) "
        "for PDE surrogate modeling tasks.\n\n"
        f"Task description:\n{task_desc}\n\n"
        "We ran several FNO configurations and measured their performance:\n"
        f"{candidates_desc}\n\n"
        "Objective:\n"
        f"{weights_desc}\n\n"
        "Question:\n"
        "1. Which configuration is the best under this objective profile?\n"
        "2. Justify your choice by explaining the trade-offs between accuracy (eval loss), "
        "model size (parameter count), and speed (avg epoch time).\n"
        "3. Then restate the recommended configuration in a clear, machine-readable JSON block with keys:\n"
        "   {\"label\", \"width\", \"num_layers\", \"modes_x\", \"modes_y\", \"num_params\"}.\n"
    )

    # ---- Build response ----
    resp_lines: List[str] = []
    resp_lines.append(f"Under the objective profile '{obj.name}', the best configuration is '{best.label}'.")
    resp_lines.append("")
    resp_lines.append("Reasoning:")
    resp_lines.append(
        f"- Accuracy: Its eval loss is {best.final_eval_loss:.6f}, "
        f"while the overall range across experiments is [{loss_min:.6f}, {loss_max:.6f}]."
    )
    resp_lines.append(
        f"- Model size: It uses {best.num_params:,} parameters, "
        f"compared to the parameter range [{param_min:.0f}, {param_max:.0f}]."
    )
    resp_lines.append(
        f"- Speed: It runs in {best.avg_epoch_time_s:.3f} seconds per epoch on {best.device}, "
        f"compared to the runtime range [{time_min:.3f}, {time_max:.3f}]."
    )
    resp_lines.append(
        "Given the specified weights on loss, parameters, and time, this configuration "
        "offers the best compromise according to the objective. "
        "More aggressive compression would reduce parameters and runtime further but "
        "would violate the accuracy threshold or increase the weighted score."
    )
    resp_lines.append("")
    resp_lines.append("Recommended configuration in JSON:")
    resp_lines.append("```json")
    resp_lines.append(json.dumps({
        "label": best.label,
        "width": best.width,
        "num_layers": best.num_layers,
        "modes_x": best.modes_x,
        "modes_y": best.modes_y,
        "num_params": best.num_params,
    }, indent=2))
    resp_lines.append("```")

    response_text = "\n".join(resp_lines)

    return {
        "prompt": prompt,
        "response": response_text,
    }


def main():
    experiments = load_experiments(RESULTS_FILE)
    if not experiments:
        print("[Step7] No experiments found, nothing to build.")
        return

    # Define several objective profiles
    profiles = [
        ObjectiveProfile(
            name="accuracy_first",
            description=(
                "Primarily minimize eval loss, but mildly prefer smaller and faster models. "
                "Used when accuracy is critical and energy is secondary."
            ),
            w_loss=0.7,
            w_params=0.15,
            w_time=0.15,
            max_loss_factor=1.02,  # within 2% of best loss
        ),
        ObjectiveProfile(
            name="balanced",
            description=(
                "Balance accuracy, model size, and speed equally. "
                "Used when we want a good trade-off between quality and energy."
            ),
            w_loss=0.34,
            w_params=0.33,
            w_time=0.33,
            max_loss_factor=1.05,  # within 5% of best loss
        ),
        ObjectiveProfile(
            name="energy_first",
            description=(
                "Prioritize small and fast models, while keeping accuracy within 10% "
                "of the best model. Used when energy/latency is more important than "
                "slightly better accuracy."
            ),
            w_loss=0.2,
            w_params=0.4,
            w_time=0.4,
            max_loss_factor=1.10,  # within 10% of best loss
        ),
    ]

    out_path = Path(OUT_LLM_DATA)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, str]] = []
    for obj in profiles:
        print(f"[Step7] Building example for profile: {obj.name}")
        ex = build_example_for_objective(experiments, obj)
        examples.append(ex)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[Step7] Wrote {len(examples)} LLM training examples to: {out_path.resolve()}")
    print("[Step7] Example prompt snippet from first profile:")
    print("----------------------------------------")
    print(examples[0]["prompt"][:400] + " ...")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
