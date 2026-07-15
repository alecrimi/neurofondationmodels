"""
Sundial isolated runner (thuml/sundial-base-128m).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API (from README, transformers==4.40.1):
    output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)
    # output.shape == (batch_size, num_samples, forecast_length)

We use transformers>=4.46.0 (Python 3.13 compatible). The model is loaded via
trust_remote_code=True — HuggingFace's current model card code is used.
If the custom generate() is still present we use it directly; otherwise
we fall back to _greedy_search (which works on newer transformers with shims).
"""
import sys
import types
import argparse
import pickle
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _patch_dynamic_cache():
    """Shim old DynamicCache attrs removed in transformers>=4.44."""
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length()
    except Exception:
        pass


def _greedy_search_fallback(model, seqs, horizon_len, num_samples, device):
    """
    Fallback for newer transformers that removed _greedy_search().
    Uses ReVIN normalisation to match TSGenerationMixin internals.
    """
    from transformers import LogitsProcessorList, StoppingCriteriaList
    from transformers.generation.stopping_criteria import MaxLengthCriteria

    context_len = seqs.shape[-1]
    means = seqs.mean(dim=-1, keepdim=True)
    stdev = seqs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
    inputs_norm = (seqs - means) / stdev

    attn_mask = torch.ones(1, context_len, dtype=torch.long, device=seqs.device)
    stopping  = StoppingCriteriaList([MaxLengthCriteria(max_length=context_len + horizon_len)])

    if not hasattr(model, "_extract_past_from_model_output"):
        def _ep(self_inner, outputs, standardize_cache_format=False, **kwargs):
            return getattr(outputs, "past_key_values", None)
        model._extract_past_from_model_output = types.MethodType(_ep, model)

    with torch.no_grad():
        output = model._greedy_search(
            input_ids=inputs_norm,
            logits_processor=LogitsProcessorList(),
            stopping_criteria=stopping,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            output_logits=False,
            return_dict_in_generate=False,
            synced_gpus=False,
            streamer=None,
            attention_mask=attn_mask,
            num_samples=num_samples,
        )

    # output: (1, num_samples, horizon_len) — reverse ReVIN
    stdev_e = stdev.unsqueeze(1).repeat(1, num_samples, 1)
    means_e = means.unsqueeze(1).repeat(1, num_samples, 1)
    return (output * stdev_e) + means_e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    device      = data["device"]
    horizon_len = data["horizon_len"]
    subjects    = data["subjects"]

    _patch_dynamic_cache()

    print(f"[Sundial] Loading thuml/sundial-base-128m on {device} …")
    model = AutoModelForCausalLM.from_pretrained(
        "thuml/sundial-base-128m",
        trust_remote_code=True,
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    print("[Sundial] Model ready.")

    # Probe which generate path works with the installed transformers version
    _use_custom_generate = callable(getattr(model, "generate", None))
    _has_greedy_search   = hasattr(model, "_greedy_search")
    print(f"[Sundial] custom generate={_use_custom_generate}, _greedy_search={_has_greedy_search}")

    num_samples = 20
    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[Sundial] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx  = win["context"].astype(np.float32)
                seqs = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
                if device != "cpu":
                    seqs = seqs.to(device)

                try:
                    # Creator's API from README — works when HuggingFace model
                    # code overrides generate() (transformers==4.40.1 era)
                    with torch.no_grad():
                        output = model.generate(
                            seqs,
                            max_new_tokens=horizon_len,
                            num_samples=num_samples,
                        )
                    # shape: (batch, num_samples, H)
                    arr = output[0].cpu().numpy()
                except (TypeError, AttributeError, RuntimeError):
                    # Fallback: _greedy_search for newer transformers
                    output = _greedy_search_fallback(model, seqs, horizon_len, num_samples, device)
                    arr = output[0].cpu().numpy()

                pred = np.median(arr, axis=0)  # (H,)
                preds.append(pred)
                targets.append(win["target"])

            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[Sundial] Done.")


if __name__ == "__main__":
    main()
