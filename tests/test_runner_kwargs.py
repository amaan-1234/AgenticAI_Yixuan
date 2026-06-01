"""Per-model runner-kwargs resolution: global -> nested runner: -> top-level lift."""

from run.run_vision_inference import _resolve_runner_kwargs


GLOBAL = {"max_model_len": 4096, "gpu_memory_utilization": 0.92,
          "max_num_seqs": 4, "enforce_eager": True}


def test_lifts_max_model_len_from_top_level():
    """Top-level max_model_len on the model entry overrides the global default."""
    entry = {"key": "qwen2_vl_2b", "id": "Qwen/Qwen2-VL-2B-Instruct", "max_model_len": 1024}
    kw = _resolve_runner_kwargs(GLOBAL, entry)
    assert kw["max_model_len"] == 1024
    # other defaults are still inherited from the global runner block
    assert kw["gpu_memory_utilization"] == 0.92
    assert kw["max_num_seqs"] == 4


def test_top_level_overrides_nested_runner():
    """Top-level lift wins over the per-model nested runner: block."""
    entry = {
        "key": "x", "id": "x",
        "max_model_len": 1024,
        "runner": {"max_model_len": 8192, "max_num_seqs": 2},
    }
    kw = _resolve_runner_kwargs(GLOBAL, entry)
    assert kw["max_model_len"] == 1024     # top-level wins
    assert kw["max_num_seqs"] == 2         # nested still applied where no top-level


def test_default_when_no_override():
    """No per-model max_model_len keeps the global default."""
    entry = {"key": "x", "id": "x"}
    kw = _resolve_runner_kwargs(GLOBAL, entry)
    assert kw["max_model_len"] == 4096


def test_quantization_still_lifts():
    """Regression: quantization continues to lift from the top level (existing behaviour)."""
    entry = {"key": "x", "id": "x", "quantization": "awq"}
    kw = _resolve_runner_kwargs(GLOBAL, entry)
    assert kw["quantization"] == "awq"
    assert kw["max_model_len"] == 4096      # untouched


def test_dtype_lifts_from_top_level():
    """Per-model dtype overrides the global default. Required for the hpc profile's
    fp16 InternVL2-8B entry where vLLM otherwise picks bfloat16 and crashes on the
    fp16-saved weights."""
    entry = {"key": "internvl2_8b", "id": "OpenGVLab/InternVL2-8B",
             "quantization": None, "dtype": "float16"}
    kw = _resolve_runner_kwargs(GLOBAL, entry)
    assert kw["dtype"] == "float16"
    assert kw["quantization"] is None
    assert kw["max_model_len"] == 4096      # other globals untouched
