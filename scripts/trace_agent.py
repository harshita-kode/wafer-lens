"""Run the agent on one wafer and print each step it took.

  python -m scripts.trace_agent samples/loc.npy
  python -m scripts.trace_agent samples/edge-ring.npy --no-report
"""
import argparse
from pathlib import Path

import numpy as np

from src.agent.graph import get_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="A 2D .npy wafer map")
    parser.add_argument("--no-report", action="store_true", help="Only print the steps")
    args = parser.parse_args()

    result = get_agent().analyze(np.load(args.file, allow_pickle=False))

    print(f"Agent steps for {args.file.name}\n")
    for step in result.agent_steps:
        print(f"  {step['node']:12s} {step['ms']:>8.1f} ms   {step['summary']}")
    print(f"\n  {'total':12s} {result.timings_ms['total']:>8.1f} ms")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if not args.no_report:
        print("\n" + "-" * 70 + "\n")
        print(result.report_markdown)


if __name__ == "__main__":
    main()