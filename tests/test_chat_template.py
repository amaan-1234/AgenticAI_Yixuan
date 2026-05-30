"""Per-family chat-template handling for the VLM ensemble (no GPU, no vLLM)."""

from unittest.mock import MagicMock

import pytest

from cac.models.vllm_runner import VLMRunner, _build_user_messages, _family


# --- _family detection --------------------------------------------------------

@pytest.mark.parametrize("model_id, expected", [
    ("Qwen/Qwen2-VL-2B-Instruct", "qwen2_vl"),
    ("Qwen/Qwen2-VL-7B-Instruct-AWQ", "qwen2_vl"),
    ("OpenGVLab/InternVL2-2B", "internvl2"),
    ("OpenGVLab/InternVL2-8B-AWQ", "internvl2"),
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
