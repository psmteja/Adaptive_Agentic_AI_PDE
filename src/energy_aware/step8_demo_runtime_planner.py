"""
energy_fno_planner.py

Runtime "planner" stub for energy-efficient FNO design.

- Loads FNO experiment logs (from step4)
- Supports several objective profiles (accuracy-first, balanced, energy-first)
- Returns a recommended configuration (label + hyperparams + JSON)

In the future, this logic can be replaced or augmented by an LLM planner.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    max_loss_factor: float = 1.0  # 1.0: no constraint; e.g. 1.05: within 5% of best


@dataclass
class PlannerResult:
    profile: ObjectiveProfile
    best_experiment: ExperimentRecord
    score: float
    loss_range: (float, float)
    params_range: (float, float)
    time_range: (float, float)

    def to_json_config(self) -> Dict[str, Any]:
        """
        JSON-ready config for use by a training script:
          { "label", "width", "num_layers", "modes_x", "modes_y", "num_params" }
        """
        e = self.best_experiment
        return {
            "label": e.label,
            "width": e.width,
            "num_layers": e.num_layers,
            "modes_x": e.modes_x,
            "modes_y": e.modes_y,
            "num_params": e.num_params,
        }


class EnergyFNOPlanner:
    def __init__(self, results_file: str = "results/poseidon_fno_experiments.jsonl"):
        self.results_file = Path(results_file)
        self.experiments: List[ExperimentRecord] = self._load_experiments()

    # ---------- loading & helpers ----------

    def _load_experiments(self) -> List[ExperimentRecord]:
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file.resolve()}")

        records: List[ExperimentRecord] = []
        with self.results_file.open("r", encoding="utf-8") as f:
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

        print(f"[Planner] Loaded {len(records)} experiments from {self.results_file.resolve()}")
        return records

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [0.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    # ---------- main planning API ----------

    def recommend(
        self,
        profile: ObjectiveProfile,
        dataset_filter: Optional[Dict[str, Any]] = None,
    ) -> PlannerResult:
        """
        Recommend a configuration under a given objective profile.

        Optionally restrict to experiments matching a dataset_filter on dataset_meta.
        Example filter:
            { "model_size": "T", "image_size": 128 }
        """

        if not self.experiments:
            raise ValueError("No experiments available for planning.")

        # Filter by dataset_meta if requested
        exps = self.experiments
        if dataset_filter:
            def matches(e: ExperimentRecord) -> bool:
                for k, v in dataset_filter.items():
                    if e.dataset_meta.get(k) != v:
                        return False
                return True
            exps = [e for e in exps if matches(e)]
            if not exps:
                raise ValueError(f"No experiments match dataset_filter={dataset_filter}")

        # Raw metrics (on filtered set)
        losses = [e.final_eval_loss for e in exps]
        params = [float(e.num_params) for e in exps]
        times = [e.avg_epoch_time_s for e in exps]

        # Accuracy constraint: only keep configs within max_loss_factor of best
        best_loss = min(losses)
        max_allowed_loss = best_loss * profile.max_loss_factor
        candidate_indices = [
            i for i, e in enumerate(exps) if e.final_eval_loss <= max_allowed_loss
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(exps)))  # fallback

        cand_exps = [exps[i] for i in candidate_indices]
        cand_losses = [losses[i] for i in candidate_indices]
        cand_params = [params[i] for i in candidate_indices]
        cand_times = [times[i] for i in candidate_indices]

        loss_norm = self._normalize(cand_losses)
        params_norm = self._normalize(cand_params)
        time_norm = self._normalize(cand_times)

        # Combined score
        scores = []
        for ln, pn, tn in zip(loss_norm, params_norm, time_norm):
            score = profile.w_loss * ln + profile.w_params * pn + profile.w_time * tn
            scores.append(score)

        best_idx_local = min(range(len(cand_exps)), key=lambda i: scores[i])
        best = cand_exps[best_idx_local]
        best_score = scores[best_idx_local]

        # Ranges for explanation
        loss_min, loss_max = min(losses), max(losses)
        param_min, param_max = min(params), max(params)
        time_min, time_max = min(times), max(times)

        return PlannerResult(
            profile=profile,
            best_experiment=best,
            score=best_score,
            loss_range=(loss_min, loss_max),
            params_range=(param_min, param_max),
            time_range=(time_min, time_max),
        )

    # ---------- helper: predefined profiles ----------

    @staticmethod
    def default_profiles() -> Dict[str, ObjectiveProfile]:
        return {
            "accuracy_first": ObjectiveProfile(
                name="accuracy_first",
                description="Prioritize accuracy; lightly prefer smaller/faster models.",
                w_loss=0.7,
                w_params=0.15,
                w_time=0.15,
                max_loss_factor=1.02,  # within 2% of best
            ),
            "balanced": ObjectiveProfile(
                name="balanced",
                description="Balance accuracy, model size, and speed.",
                w_loss=0.34,
                w_params=0.33,
                w_time=0.33,
                max_loss_factor=1.05,  # within 5% of best
            ),
            "energy_first": ObjectiveProfile(
                name="energy_first",
                description="Prioritize small & fast models; accuracy can be up to 10% worse than best.",
                w_loss=0.2,
                w_params=0.4,
                w_time=0.4,
                max_loss_factor=1.10,  # within 10% of best
            ),
        }


"""
Step 8: Demo of the runtime EnergyFNOPlanner.

Run:
    python step8_demo_runtime_planner.py
"""



def main():
    planner = EnergyFNOPlanner("results/poseidon_fno_experiments.jsonl")
    profiles = planner.default_profiles()

    # Example: filter to a specific dataset (here it's all Poseidon-T 128x128)
    dataset_filter = {
        "model_size": "T",
        "image_size": 128,
        "num_in_channels": 4,
        "num_out_channels": 4,
    }

    for name, profile in profiles.items():
        print("\n========================================")
        print(f"[Step8] Objective profile: {name}")
        print(f"  Description: {profile.description}")
        result = planner.recommend(profile, dataset_filter=dataset_filter)
        e = result.best_experiment

        print(f"  -> Recommended config label: {e.label}")
        print(f"     width        : {e.width}")
        print(f"     num_layers   : {e.num_layers}")
        print(f"     modes_x/y    : {e.modes_x}/{e.modes_y}")
        print(f"     num_params   : {e.num_params:,}")
        print(f"     final_loss   : {e.final_eval_loss:.6f}")
        print(f"     avg_epoch_s  : {e.avg_epoch_time_s:.3f}")
        print(f"     score        : {result.score:.3f}")
        print(f"     loss_range   : {result.loss_range}")
        print(f"     params_range : {result.params_range}")
        print(f"     time_range   : {result.time_range}")

        json_cfg = result.to_json_config()
        print("  JSON-ready config:")
        print("   ", json_cfg)

    print("\n[Step8] Demo complete ✅")


if __name__ == "__main__":
    main()
