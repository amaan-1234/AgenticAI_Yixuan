"""Entry point: compute real MTA from model rationales (Phase 1.3).

Two modes:
  python -m run.compute_mta --sanity        # acceptance test on hand-built pairs (no inference needed)
  python -m run.compute_mta                 # compute MTA from outputs/raw/*.jsonl rationales

The --sanity mode is the 1.3 acceptance check: paraphrase pairs must score higher
than contradictory pairs. It runs today without any VLM outputs.
"""

import argparse

import numpy as np

from cac.mta.cross_encoder import MTAScorer, compute_and_cache


SANITY_AGREE = [
    ("This is a small bird perched on a tree branch.",
     "A little bird is sitting on the branch of a tree."),
    ("The animal has four legs and pointed ears, likely a cat.",
     "It looks like a cat: four legs, small body, pointed ears."),
    ("A large commercial airplane flying in a clear blue sky.",
     "There is a jet aircraft up in the sky."),
]
SANITY_DISAGREE = [
    ("This is clearly an airplane against the sky.",
     "This is a green frog sitting on a leaf."),
    ("The image shows a red automobile on a road.",
     "The image shows a deer standing in a forest."),
    ("A ship sailing on the open ocean.",
     "A small brown horse in a grassy field."),
]


def sanity():
    scorer = MTAScorer()
    agree = scorer.pair_scores(SANITY_AGREE)
    disagree = scorer.pair_scores(SANITY_DISAGREE)
    print(f"[sanity] agree   scores: {np.round(agree, 3)}  mean={agree.mean():.3f}")
    print(f"[sanity] disagree scores: {np.round(disagree, 3)}  mean={disagree.mean():.3f}")
    assert agree.mean() > disagree.mean(), (
        "cross-encoder did not rank paraphrases above contradictions"
    )
    # Also exercise the per-item MTA aggregation with a 3-rationale item.
    mta = scorer.mta_for_rationales([
        [SANITY_AGREE[0][0], SANITY_AGREE[0][1], SANITY_AGREE[1][1]],
        [SANITY_DISAGREE[0][0], SANITY_DISAGREE[0][1], SANITY_DISAGREE[1][1]],
    ])
    print(f"[sanity] per-item MTA (agree-ish, disagree-ish): {np.round(mta, 3)}")
    assert mta[0] > mta[1], "per-item MTA aggregation inconsistent"
    print("[ok] MTA cross-encoder sanity test passed.")


def from_outputs():
    from cac.ensemble.inference import load_rationales

    rationales = load_rationales()  # [N][M] strings
    mta = compute_and_cache(rationales)
    print(f"[ok] MTA computed for {len(mta)} items, mean={mta.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true",
                    help="run the acceptance sanity test instead of reading outputs")
    args = ap.parse_args()
    sanity() if args.sanity else from_outputs()


if __name__ == "__main__":
    main()
