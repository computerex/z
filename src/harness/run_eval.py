"""CLI for running evaluations."""

import asyncio
import argparse
import sys
from pathlib import Path

from .config import Config


def main():
    """Main CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Run evaluation tasks on the agentic harness"
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
        default=None,
        help="Maximum number of tasks to run"
    )
    parser.add_argument(
        "-c", "--category",
        type=str,
        nargs="+",
        choices=["bug_fix", "feature_add", "refactoring", "debugging", "testing", "code_search", "documentation"],
        help="Task categories to run"
    )
    parser.add_argument(
        "-d", "--difficulty",
        type=str,
        nargs="+",
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to run"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--swebench",
        action="store_true",
        help="Run SWE-bench evaluation instead"
    )
    parser.add_argument(
        "--swebench-dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use"
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
    
    if args.swebench:
        # Run SWE-bench evaluation
        from .swebench import run_swebench_evaluation
        
        summary = asyncio.run(run_swebench_evaluation(
            config=config,
            max_tasks=args.num_tasks or 10,
            dataset=args.swebench_dataset,
            output_dir=Path(args.output),
        ))
    else:
        # Run custom evaluation tasks
        from .evaluation import run_evaluation
        
        summary = asyncio.run(run_evaluation(
            config=config,
            categories=args.category,
            difficulties=args.difficulty,
            max_tasks=args.num_tasks,
            output_dir=Path(args.output),
        ))
    
    # Exit with appropriate code
    if summary.get("pass_rate", 0) > 0.5:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
