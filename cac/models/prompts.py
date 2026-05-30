"""Prompt templates for the VLM ensemble (CIFAR-10H).

CIFAR-10 images are 32x32; they are upscaled before being handed to the vision
encoder (see `cac/ensemble/inference.py`). The distribution prompt forces a single
letter so position-0 logprobs map cleanly onto the 10 classes.
"""

from __future__ import annotations

from cac.data.labels import CIFAR10_CLASSES, options_block

_LEGEND = options_block()  # "A=airplane, B=automobile, ..."

DIST_PROMPT = (
    "This is a low-resolution (32x32) photo from the CIFAR-10 dataset, upscaled. "
    "Classify the main object into exactly one category.\n"
    f"Options: {_LEGEND}.\n"
    "Answer with ONLY the single capital letter of the best category. No other text."
)

RATIONALE_PROMPT = (
    "This is a low-resolution (32x32) CIFAR-10 photo, upscaled. "
    f"Classify into one of: {', '.join(CIFAR10_CLASSES)}.\n"
    "Return JSON with EXACTLY these keys:\n"
    '  "label": <class string>,\n'
    '  "rationale": {\n'
    '    "key_features": [up to 5 short visual cues seen in the image],\n'
    '    "reasoning": one sentence connecting features to the class,\n'
    '    "conclusion": one sentence stating the chosen class,\n'
    '    "confidence_qualifier": one of "high", "medium", "low"\n'
    "  }\n"
    "Keep each string under 25 words."
)
