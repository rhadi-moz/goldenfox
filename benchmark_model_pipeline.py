from generator import run_benchmark_generation
from evaluator import run_benchmark_evaluation
from src.utils import load_config
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Goldenfox evaluation benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_run_config.yaml",
        help="Path to the YAML config file",
    )

    parser.add_argument(
    "--model",
    type=str,
    help="Optional override for model name",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Override config with CLI args if present
    if args.model:
        config["run"]["model_name"] = args.model

    # Run generation if config allows
    generation_flags = config.get("benchmark", {}).get("generation", {})
    if any(generation_flags.values()):
        print("Running generation...")
        run_benchmark_generation.main(["--config", args.config])
    else:
        print("Skipping generation.")

    # Run evaluation if config allows
    evaluation_flags = config.get("benchmark", {}).get("evaluation", {})
    if any(evaluation_flags.values()):
        print("Running evaluation...")
        run_benchmark_evaluation.main(["--config", args.config])
    else:
        print("Skipping evaluation.")

if __name__ == "__main__":
    main()
