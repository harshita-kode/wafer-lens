"""Example client: call a running WaferLens API for every sample wafer.

Start the server first:  uvicorn src.api.main:app
Then run:                python -m scripts.api_client
"""
import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--show-report", help="Print the full report for this sample")
    args = parser.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"Health: {health['status']}  (llm={health['llm']}, similar={health['similar_index']})\n")

    samples = requests.get(f"{args.url}/samples", timeout=10).json()["samples"]
    print(f"{'sample':12s} {'predicted':12s} {'conf':>6s} {'uncertain':>9s} {'source':>8s} {'ms':>8s}")

    for name in samples:
        wafer = requests.get(f"{args.url}/samples/{name}", timeout=10).json()["wafer_map"]
        response = requests.post(f"{args.url}/analyze", json={"wafer_map": wafer}, timeout=120)
        response.raise_for_status()
        result = response.json()

        p = result["prediction"]
        print(f"{name:12s} {p['label']:12s} {p['confidence']:6.0%} {str(p['uncertain']):>9s} "
              f"{result['generated_by']:>8s} {response.headers['X-Process-Time-Ms']:>8s}")

        if name == args.show_report:
            print("\n" + result["report_markdown"] + "\n")

    bad = requests.post(f"{args.url}/analyze", json={"wafer_map": [[7] * 20] * 20}, timeout=10)
    print(f"\nInvalid wafer -> {bad.status_code}: {bad.json()['detail']}")


if __name__ == "__main__":
    main()