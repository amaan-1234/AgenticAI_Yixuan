"""vLLM offline runner for a single vision-language model.

Runs on Linux+CUDA (WSL2 locally, or the HPC). Not importable-and-usable on the
Windows host (vLLM is Linux-only) — the rest of the pipeline does not import this
module unless you actually run inference, so the repo stays usable on Windows.

Per image we issue two calls:
  1. distribution call  — single-token answer (A..J) with logprobs -> soft labels
  2. rationale call      — guided JSON {label, rationale} for MTA + the 'valid JSON' gate

Model-specific prompt formatting is delegated to each model's HF chat template via
AutoProcessor.apply_chat_template, which is the portable recipe across
Qwen2.5-VL / LLaVA-1.6 / InternVL2 in recent vLLM. If a model needs special
handling, that is the single place to adjust — verify on the 100-image batch first.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cac.models import prompts, schema

UPSCALE = 224  # CIFAR-10 is 32x32; upscale before the vision encoder


def _family(model_id: str) -> str:
    """Identify the chat-template family from a HF model id (case-insensitive substring).

    Returns one of 'qwen2_vl', 'internvl2', 'phi3v', 'llava', 'unknown'. The model
    ids in config/models_vision.yaml are canonical, so substring matching is
    reliable here. Phi3V requires both 'phi-3' and 'vision' so text-only Phi-3
    checkpoints (e.g. Phi-3-mini-4k-instruct) don't match.
    """
    m = model_id.lower()
    if "internvl2" in m:
        return "internvl2"
    if "qwen2-vl" in m or "qwen2_vl" in m:
        return "qwen2_vl"
    if "phi-3" in m and "vision" in m:
        return "phi3v"
    if "llava" in m:
        return "llava"
    return "unknown"


def _build_user_messages(model_id: str, instruction: str) -> list[dict]:
    """Build the messages list expected by each family's chat template.

    InternVL2 concatenates `content` as a string in its template (and a list-of-parts
    triggers `TypeError: can only concatenate str (not 'list') to str`), so InternVL2
    gets string content with an embedded `<image>` placeholder. Phi3V uses the same
    string-content shape but with `<|image_1|>` as its image placeholder. Qwen2-VL
    and LLaVA iterate over a list of content parts.
    """
    fam = _family(model_id)
    if fam == "internvl2":
        return [{"role": "user", "content": f"<image>\n{instruction}"}]
    if fam == "phi3v":
        return [{"role": "user", "content": f"<|image_1|>\n{instruction}"}]
    return [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }]


class VLMRunner:
    def __init__(self, model_id: str, *, quantization: str | None = "awq",
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.92,
                 max_num_seqs: int = 4, enforce_eager: bool = True,
                 trust_remote_code: bool = True, dtype: str = "auto"):
        from transformers import AutoProcessor
        from vllm import LLM

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        print(f"[vllm] init {model_id}: max_model_len={max_model_len} "
              f"max_num_seqs={max_num_seqs} quantization={quantization} "
              f"gpu_mem={gpu_memory_utilization}")
        self.llm = LLM(
            model=model_id,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            limit_mm_per_prompt={"image": 1},
            # lm-format-enforcer validates JSON per generated token instead of
            # precompiling a ~4.7k-state FSM (outlines' default), which can take hours on
            # memory-constrained WSL2. Shipped with vLLM 0.6.3.post1; schema unchanged.
            guided_decoding_backend="lm-format-enforcer",
        )

    # ---- prompt construction -------------------------------------------------
    def _chat_text(self, instruction: str) -> str:
        messages = _build_user_messages(self.model_id, instruction)
        # Phi3VProcessor doesn't expose chat_template on the processor itself —
        # it lives on the wrapped tokenizer. Other families (Qwen2-VL, InternVL2,
        # LLaVA) keep the processor-level path that AutoProcessor wires up.
        if _family(self.model_id) == "phi3v":
            return self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _prep_image(img: np.ndarray) -> Image.Image:
        return Image.fromarray(img).convert("RGB").resize((UPSCALE, UPSCALE), Image.BICUBIC)

    def _inputs(self, instruction: str, images: list[np.ndarray]) -> list[dict]:
        text = self._chat_text(instruction)
        return [{"prompt": text, "multi_modal_data": {"image": self._prep_image(im)}}
                for im in images]

    # ---- the two calls -------------------------------------------------------
    def distribution_call(self, images: list[np.ndarray], top_logprobs: int = 20):
        """Return list of {letter: logprob} dicts (one per image) from position-0 logprobs."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_logprobs)
        outs = self.llm.generate(self._inputs(prompts.DIST_PROMPT, images), sp)
        results = []
        for o in outs:
            pos0 = o.outputs[0].logprobs[0]  # {token_id: Logprob(logprob, decoded_token,...)}
            letter_lp: dict[str, float] = {}
            for lp in pos0.values():
                tok = (lp.decoded_token or "").strip().upper()
                if len(tok) == 1 and "A" <= tok <= "J":
                    letter_lp[tok] = max(letter_lp.get(tok, -1e9), lp.logprob)
            results.append(letter_lp)
        return results

    def rationale_call(self, images: list[np.ndarray], max_tokens: int = 512):
        """Return list of {'label': str, 'rationale': str} via guided JSON."""
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        guided = GuidedDecodingParams(json=schema.label_json_schema())
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=guided)
        outs = self.llm.generate(self._inputs(prompts.RATIONALE_PROMPT, images), sp)
        from cac.models.json_recovery import parse_rationale_response

        results = []
        for o in outs:
            txt = o.outputs[0].text
            obj, status = parse_rationale_response(txt)
            rationale = obj.get("rationale", {})
            results.append({
                "label": str(obj.get("label", "")),
                "rationale": rationale if isinstance(rationale, dict) else {},
                "_raw": txt,
                "parse_status": status,
            })
        return results

    def close(self):
        """Release GPU memory so the next model can load (sequential swap on 8 GB)."""
        import gc

        import torch

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
