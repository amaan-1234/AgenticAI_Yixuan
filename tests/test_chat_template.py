"""Per-family chat-template handling for the VLM ensemble (no GPU, no vLLM)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cac.models.vllm_runner import (
    IMAGE_INPUT_SIZE_BY_FAMILY,
    VLMRunner,
    _MM_PROCESSOR_KWARGS_BY_FAMILY,
    _build_user_messages,
    _family,
)


# --- _family detection --------------------------------------------------------

@pytest.mark.parametrize("model_id, expected", [
    ("Qwen/Qwen2-VL-2B-Instruct", "qwen2_vl"),
    ("Qwen/Qwen2-VL-7B-Instruct-AWQ", "qwen2_vl"),
    ("OpenGVLab/InternVL2-2B", "internvl2"),
    ("OpenGVLab/InternVL2-8B-AWQ", "internvl2"),
    ("microsoft/Phi-3.5-vision-instruct", "phi3v"),
    ("microsoft/Phi-3-vision-128k-instruct", "phi3v"),
    # text-only Phi-3 must not match phi3v (no 'vision' in id)
    ("microsoft/Phi-3-mini-4k-instruct", "unknown"),
    ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "llava"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", "llava"),
    ("llava-hf/llava-1.5-7b-hf", "llava"),
    ("some/other-model", "unknown"),
])
def test_family_detection(model_id, expected):
    assert _family(model_id) == expected


# --- _build_user_messages shape per family -----------------------------------

def test_internvl2_uses_string_content_with_image_placeholder():
    msgs = _build_user_messages("OpenGVLab/InternVL2-8B-AWQ", "classify this image")
    assert isinstance(msgs[0]["content"], str)
    assert "<image>" in msgs[0]["content"]
    assert "classify this image" in msgs[0]["content"]


def test_phi3v_uses_string_content_with_image_1_placeholder():
    msgs = _build_user_messages("microsoft/Phi-3.5-vision-instruct", "classify this image")
    assert isinstance(msgs[0]["content"], str)
    assert "<|image_1|>" in msgs[0]["content"]
    assert "classify this image" in msgs[0]["content"]
    # phi3v must NOT reuse InternVL's <image> placeholder
    assert "<image>" not in msgs[0]["content"].replace("<|image_1|>", "")


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
])
def test_list_format_for_qwen_and_llava(model_id):
    msgs = _build_user_messages(model_id, "classify this image")
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert {"type": "image"} in content
    assert any(p.get("type") == "text" and p["text"] == "classify this image" for p in content)


def test_unknown_family_defaults_to_list_format():
    msgs = _build_user_messages("some/other-model", "x")
    assert isinstance(msgs[0]["content"], list)


# --- _chat_text with a mocked apply_chat_template (one per family) -----------

def _mock_runner(model_id: str) -> VLMRunner:
    """Construct a VLMRunner without running its heavy __init__ (no vLLM, no GPU)."""
    r = object.__new__(VLMRunner)
    r.model_id = model_id
    r.processor = MagicMock()
    r.processor.apply_chat_template.return_value = "RENDERED"
    return r


def test_chat_text_qwen2vl_passes_list_content_to_template():
    r = _mock_runner("Qwen/Qwen2-VL-7B-Instruct-AWQ")
    assert r._chat_text("classify") == "RENDERED"
    args, kwargs = r.processor.apply_chat_template.call_args
    messages = args[0]
    assert isinstance(messages[0]["content"], list)
    assert kwargs.get("add_generation_prompt") is True


def test_chat_text_internvl2_passes_string_content_to_template():
    r = _mock_runner("OpenGVLab/InternVL2-8B-AWQ")
    assert r._chat_text("classify") == "RENDERED"
    args, _ = r.processor.apply_chat_template.call_args
    messages = args[0]
    assert isinstance(messages[0]["content"], str)
    assert "<image>" in messages[0]["content"]


def test_chat_text_llava_passes_list_content_to_template():
    r = _mock_runner("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    assert r._chat_text("classify") == "RENDERED"
    args, _ = r.processor.apply_chat_template.call_args
    messages = args[0]
    assert isinstance(messages[0]["content"], list)


def test_chat_text_phi3v_uses_processor_tokenizer_not_processor():
    """Phi3VProcessor has no chat_template; the template lives on the wrapped
    tokenizer. _chat_text must route to processor.tokenizer.apply_chat_template
    for phi3v and never touch the processor-level method."""
    r = _mock_runner("microsoft/Phi-3.5-vision-instruct")
    r.processor.tokenizer.apply_chat_template.return_value = "PHI3V_RENDERED"
    assert r._chat_text("classify") == "PHI3V_RENDERED"
    # processor-level apply_chat_template must NOT have been called
    assert r.processor.apply_chat_template.call_count == 0
    args, kwargs = r.processor.tokenizer.apply_chat_template.call_args
    messages = args[0]
    assert isinstance(messages[0]["content"], str)
    assert "<|image_1|>" in messages[0]["content"]
    assert kwargs.get("add_generation_prompt") is True


def test_chat_text_qwen2vl_does_not_touch_tokenizer_path():
    """Regression guard: the existing Qwen/InternVL/LLaVA path must keep using
    processor.apply_chat_template and never fall through to the tokenizer."""
    r = _mock_runner("Qwen/Qwen2-VL-7B-Instruct-AWQ")
    r._chat_text("classify")
    assert r.processor.apply_chat_template.call_count == 1
    assert r.processor.tokenizer.apply_chat_template.call_count == 0


def test_chat_text_internvl2_does_not_touch_tokenizer_path():
    r = _mock_runner("OpenGVLab/InternVL2-8B-AWQ")
    r._chat_text("classify")
    assert r.processor.apply_chat_template.call_count == 1
    assert r.processor.tokenizer.apply_chat_template.call_count == 0


# --- family-aware _prep_image resize target ----------------------------------

@pytest.mark.parametrize("model_id, expected_size", [
    ("OpenGVLab/InternVL2-8B-AWQ",                224),
    ("OpenGVLab/InternVL2-2B",                    224),
    ("Qwen/Qwen2-VL-7B-Instruct-AWQ",             224),
    ("Qwen/Qwen2-VL-2B-Instruct",                 224),
    ("microsoft/Phi-3.5-vision-instruct",         224),
    ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 224),
    ("some/other-model",                          224),
])
def test_prep_image_resize_target_by_family(model_id, expected_size):
    """All families currently resize to 224x224.

    The earlier attempt to route InternVL2 to 448 REGRESSED its mean max-prob
    (0.597 -> 0.584) because vLLM's InternVL2 plugin interpreted the 448 input
    as room for a 2x2 dynamic tile grid. InternVL2's single-tile path is now
    enforced via mm_processor_kwargs={'max_dynamic_patch': 1} instead, and the
    input stays at 224 so the plugin upscales to a single 448 native tile.
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    pil = VLMRunner._prep_image(img, model_id)
    assert pil.size == (expected_size, expected_size)
    assert pil.mode == "RGB"


def test_image_input_size_map_covers_all_known_families():
    """If a new family is added to _family, it must also appear in the size map
    or the .get() fallback will silently downscale it to the 'unknown' default."""
    assert set(IMAGE_INPUT_SIZE_BY_FAMILY) >= {
        "internvl2", "qwen2_vl", "phi3v", "llava", "unknown",
    }


def test_prep_image_unknown_family_falls_back_to_unknown_default():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    pil = VLMRunner._prep_image(img, "totally/unrecognized")
    assert pil.size == (IMAGE_INPUT_SIZE_BY_FAMILY["unknown"],
                        IMAGE_INPUT_SIZE_BY_FAMILY["unknown"])


# --- family-gated mm_processor_kwargs ----------------------------------------

@pytest.mark.parametrize("model_id, expected", [
    ("OpenGVLab/InternVL2-8B-AWQ",                {"max_dynamic_patch": 1}),
    ("OpenGVLab/InternVL2-2B",                    {"max_dynamic_patch": 1}),
    ("Qwen/Qwen2-VL-7B-Instruct-AWQ",             None),
    ("Qwen/Qwen2-VL-2B-Instruct",                 None),
    ("microsoft/Phi-3.5-vision-instruct",         None),
    ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", None),
    ("some/other-model",                          None),
])
def test_mm_processor_kwargs_by_family(model_id, expected):
    """InternVL2 must get max_dynamic_patch=1 to disable redundant tiling;
    every other family must get None so the kwarg is omitted from the LLM call
    and vLLM's defaults apply."""
    assert _MM_PROCESSOR_KWARGS_BY_FAMILY.get(_family(model_id)) == expected


def test_mm_processor_kwargs_map_only_contains_known_families():
    """Guard against typos in the map — every key must be a real _family() output
    so a misspelled family name doesn't silently fail to apply its override."""
    known = {"internvl2", "qwen2_vl", "phi3v", "llava", "unknown"}
    assert set(_MM_PROCESSOR_KWARGS_BY_FAMILY) <= known
