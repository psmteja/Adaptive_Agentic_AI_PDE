import argparse
import yaml

from agent import ExperimentConfig, ExperimentAgent


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_name=cfg_raw.get("experiment_name", "pde_experiment"),
        nx=cfg_raw["data"]["nx"],
        ny=cfg_raw["data"]["ny"],
        nt=cfg_raw["data"]["nt"],
        dt=cfg_raw["data"]["dt"],
        nu=cfg_raw["data"]["nu"],
        model_variant=cfg_raw["model"]["variant"],
        max_horizon=cfg_raw["rollout"]["max_horizon"],
        horizons_to_eval=list(cfg_raw["rollout"]["horizons_to_eval"]),
        output_dir=cfg_raw["plotting"]["output_dir"],
        horizon_for_plot=cfg_raw["plotting"]["horizon_for_plot"],
    )


def main():
    parser = argparse.ArgumentParser(description="PDE Agent Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config_example.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent = ExperimentAgent(cfg)
    metrics_by_horizon = agent.run()

    print("\n=== Summary ===")
    for h, metrics in metrics_by_horizon.items():
        print(f"Horizon {h}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()
