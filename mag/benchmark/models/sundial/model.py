"""
Sundial wrapper — thuml/sundial-base-128m via HuggingFace transformers.
Probabilistic: 20 sample paths, returns median.

Works with transformers >= 4.44 via compatibility shims for the old DynamicCache API.
"""
import numpy as np
import torch


class SundialWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        # Sundial's ts_generation_mixin uses the old DynamicCache API which was
        # progressively removed in transformers >= 4.44. Shim back everything it needs.
        try:
            from transformers.cache_utils import DynamicCache
            # seen_tokens → current sequence length
            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(
                    lambda self: self.get_seq_length()
                )
            # get_max_length() → None for dynamic cache (no fixed upper bound)
            if not hasattr(DynamicCache, "get_max_length"):
                DynamicCache.get_max_length = lambda self: None
            # get_usable_length(new_seq_len) → new_seq_len for dynamic cache
            # (all requested positions are "usable" since it grows without bound)
            if not hasattr(DynamicCache, "get_usable_length"):
                DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length()
        except Exception:
            pass

        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
        )
        self._model.eval()
        if self.device != "cpu":
            self._model = self._model.to(self.device)

        # transformers 4.57.6 removed _extract_past_from_model_output from GenerationMixin.
        # TSGenerationMixin._greedy_search still calls it, so shim it back.
        if not hasattr(self._model, "_extract_past_from_model_output"):
            import types
            def _extract_past_from_model_output(self_inner, outputs, standardize_cache_format=False, **kwargs):
                if hasattr(outputs, "past_key_values"):
                    return outputs.past_key_values
                return None
            self._model._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output, self._model
            )

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()

        seqs = torch.tensor(context_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
        if self.device != "cpu":
            seqs = seqs.to(self.device)

        # transformers 4.57.6 removed _greedy_search from the generate() dispatch
        # (replaced by _generate). Sundial's TSGenerationMixin._greedy_search is never
        # reached via model.generate(), causing tensor shape errors in the standard path.
        # We call _greedy_search directly, replicating what TSGenerationMixin.generate() does.
        from transformers import LogitsProcessorList, StoppingCriteriaList
        from transformers.generation.stopping_criteria import MaxLengthCriteria

        num_samples = 20
        context_len = seqs.shape[1]
        max_length = context_len + horizon_len  # stopping criterion for _greedy_search

        # ReVIN normalisation (TSGenerationMixin.generate does this before calling super)
        means = seqs.mean(dim=-1, keepdim=True)
        stdev = seqs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        inputs_norm = (seqs - means) / stdev

        attention_mask = torch.ones(1, context_len, dtype=torch.long, device=seqs.device)
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        with torch.no_grad():
            output = self._model._greedy_search(
                input_ids=inputs_norm,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=stopping_criteria,
                max_length=None,       # deprecated param; stopping_criteria used instead
                pad_token_id=None,
                eos_token_id=None,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
                output_logits=False,
                return_dict_in_generate=False,
                synced_gpus=False,
                streamer=None,
                attention_mask=attention_mask,
                num_samples=num_samples,
            )

        # output shape: (1, num_samples, horizon_len) — _greedy_search already truncates
        # Reverse ReVIN (TSGenerationMixin.generate does this after super().generate)
        stdev_exp = stdev.unsqueeze(1).repeat(1, num_samples, 1)
        means_exp = means.unsqueeze(1).repeat(1, num_samples, 1)
        output = (output * stdev_exp) + means_exp

        arr = output[0].cpu().numpy()  # (num_samples, horizon_len)
        return np.median(arr, axis=0)  # (horizon_len,)
