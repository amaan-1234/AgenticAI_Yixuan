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
