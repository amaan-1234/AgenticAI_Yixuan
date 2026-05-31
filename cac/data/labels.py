"""CIFAR-10 class names and the single-token letter map used for soft labels.

The ensemble extracts soft-label distributions from token logprobs (see
`cac/ensemble/soft_labels.py`). To guarantee each class maps to exactly one
token across tokenizers, we ask the model to answer with a single uppercase
letter A..J rather than the multi-token class word. `verify_single_token`
checks this assumption per model before inference.
"""

from __future__ import annotations

# Canonical CIFAR-10 test-set order (matches torchvision and cifar10h-probs.npy).
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Class index -> answer letter. A..J are single tokens in Llama/Qwen/Mistral/etc.
LETTERS = [chr(ord("A") + i) for i in range(len(CIFAR10_CLASSES))]  # A..J
LETTER_TO_IDX = {ltr: i for i, ltr in enumerate(LETTERS)}
IDX_TO_CLASS = dict(enumerate(CIFAR10_CLASSES))


def options_block() -> str:
    """Render the 'A=airplane, B=automobile, ...' legend for prompts."""
    return ", ".join(f"{ltr}={cls}" for ltr, cls in zip(LETTERS, CIFAR10_CLASSES))


def verify_single_token(tokenizer, letters=LETTERS) -> dict[str, list[int]]:
    """Return {letter: token_ids}. Caller should assert every value has len==1.

    Some tokenizers encode a leading-space variant; we test the bare letter,
    which is what guided single-token decoding emits at position 0.
    """
    out = {}
    for ltr in letters:
        ids = tokenizer.encode(ltr, add_special_tokens=False)
        out[ltr] = ids
    return out
