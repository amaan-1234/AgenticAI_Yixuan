"""vLLM offline runner for a single TEXT LLM on NLI (analogue of vllm_runner.py).

Two calls per (premise, hypothesis), mirroring the VLM contract so the rest of the
pipeline is reused unchanged:
  1. distribution_call -> {letter: logprob} over A/B/C from position-0 logprobs
  2. rationale_call    -> guided JSON {label, rationale{...}}

Family-aware prompting: gemma-2 has NO system role (its chat template rejects one),
so all instruction goes in the user turn. Qwen accepts the same shape, so we use a
single user-turn format for both.
"""
from __future__ import annotations

from cac.data.labels_nli import options_block, LETTERS
from cac.models import schema_nli

_DIST_INSTRUCTION = (
    "You are given a premise and a hypothesis. Decide the relationship.\n"
    "Options: {opts}.\n"
    "Premise: {premise}\nHypothesis: {hypothesis}\n"
    "Answer with ONLY the single letter ({letters}). Answer:"
)
_RATIONALE_INSTRUCTION = (
    "You are given a premise and a hypothesis. Decide whether the hypothesis is "
    "entailment, neutral, or contradiction with respect to the premise.\n"
    "Premise: {premise}\nHypothesis: {hypothesis}\n"
    "Respond with a JSON object containing your label and a structured rationale."
)


def _family(model_id: str) -> str:
    m = model_id.lower()
    if "gemma" in m:
        return "gemma"
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    if "mistral" in m:
        return "mistral"
    return "unknown"


class TextRunner:
    def __init__(self, model_id: str, *, quantization: str | None = None,
                 max_model_len: int = 2048, gpu_memory_utilization: float = 0.90,
                 max_num_seqs: int = 16, enforce_eager: bool = True,
                 trust_remote_code: bool = True, dtype: str = "auto"):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_id = model_id
        self.family = _family(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        print(f"[vllm-text] init {model_id} (family={self.family}) "
              f"max_model_len={max_model_len} quant={quantization} dtype={dtype}")
        self.llm = LLM(
            model=model_id, quantization=quantization, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager, trust_remote_code=trust_remote_code,
            dtype=dtype, guided_decoding_backend="lm-format-enforcer",
        )

    def _chat(self, instruction: str) -> str:
        msgs = [{"role": "user", "content": instruction}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def _prompts(self, template, pairs):
        out = []
        for premise, hypothesis in pairs:
            instr = template.format(opts=options_block(),
                                    letters="/".join(LETTERS),
                                    premise=premise, hypothesis=hypothesis)
            out.append(self._chat(instr))
        return out

    def distribution_call(self, pairs, top_logprobs: int = 20):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_logprobs)
        outs = self.llm.generate(self._prompts(_DIST_INSTRUCTION, pairs), sp)
        results = []
        for o in outs:
            pos0 = o.outputs[0].logprobs[0]
            letter_lp = {}
            for lp in pos0.values():
                tok = (lp.decoded_token or "").strip().upper()
                if tok in ("A", "B", "C"):
                    letter_lp[tok] = max(letter_lp.get(tok, -1e9), lp.logprob)
            results.append(letter_lp)
        return results

    def rationale_call(self, pairs, max_tokens: int = 512):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from cac.models.json_recovery import parse_rationale_response
        guided = GuidedDecodingParams(json=schema_nli.label_json_schema())
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=guided)
        outs = self.llm.generate(self._prompts(_RATIONALE_INSTRUCTION, pairs), sp)
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
        import gc, torch
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
