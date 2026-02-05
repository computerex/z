"""CLI for running SWE-bench evaluation."""

import asyncio
import argparse
import sys
from pathlib import Path

from .config import Config
from .swebench import run_swebench_evaluation


def main():
    """Main CLI entry point for SWE-bench evaluation."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation on the agentic harness"
    )
    parser.add_argument(
        "-e", "--env",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    parser.add_argument(
        "-n", "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks to run (default: 10)"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=[
            "princeton-nlp/SWE-bench",
            "princeton-nlp/SWE-bench_Lite",
            "princeton-nlp/SWE-bench_Verified",
        ],
        help="SWE-bench dataset to use"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="swebench_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "-i", "--instance-ids",
        type=str,
        nargs="+",
        help="Specific instance IDs to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Load config
    env_path = Path(args.env)
    if env_path.exists():
        config = Config.from_env(env_path)
    else:
        config = Config.from_env()
    
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    summary = asyncio.run(run_swebench_evaluation(
        config=config,
        max_tasks=args.num_tasks,
        dataset=args.dataset,
        output_dir=Path(args.output),
        instance_ids=args.instance_ids,
    ))
    
    # Exit with appropriate code
    if summary["resolution_rate"] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
