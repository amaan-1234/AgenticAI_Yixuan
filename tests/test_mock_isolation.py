"""Mock fixtures must not contaminate real-output loaders (no GPU)."""

import json

from cac import config
from cac.ensemble import inference


def test_default_discovery_skips_mocks_subdir():
    """A file at outputs/raw/mocks/foo.jsonl is invisible to real loaders."""
    real_path = config.RAW_DIR / "_iso_real.jsonl"
    mock_path = config.MOCK_RAW_DIR / "_iso_mock.jsonl"
    real_path.write_text(json.dumps({"idx": 0}) + "\n", encoding="utf-8")
    mock_path.write_text(json.dumps({"idx": 0}) + "\n", encoding="utf-8")
    try:
        real_keys = inference.model_keys_from_outputs()           # default mock=False
        mock_keys = inference.model_keys_from_outputs(mock=True)
        assert "_iso_real" in real_keys
        assert "_iso_mock" not in real_keys, "mocks leaked into real discovery"
        assert "_iso_mock" in mock_keys
        assert "_iso_real" not in mock_keys, "real leaked into mock discovery"
    finally:
        real_path.unlink(missing_ok=True)
        mock_path.unlink(missing_ok=True)


def test_raw_path_routes_by_mock_flag():
    assert inference._raw_path("x") == config.RAW_DIR / "x.jsonl"
    assert inference._raw_path("x", mock=True) == config.MOCK_RAW_DIR / "x.jsonl"
