"""Entry point: build the consolidated results DB (Ask 4).

    python -m run.build_results

Reads outputs/raw/*.jsonl (real or fixture), joins with JSD / human entropy / the
calibrated escalation decision, and writes outputs/results.jsonl (one row per
image-model, written incrementally).
"""

from cac.pipeline.results import build_results


def main():
    build_results()


if __name__ == "__main__":
    main()
