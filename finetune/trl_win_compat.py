"""Fallback pure-torch implementation of TRL 1.15's chunked log-prob kernel.

TRL's SFTTrainer unconditionally patches the model with a "fused LM head"
that requires `triton` (Linux-only wheels). On Windows the kernel import
fails and `trl.trainer.utils._ChunkedLogProbFunction` becomes None, which
crashes the first training step. This module provides an equivalent,
differentiable torch-only implementation that can be patched in.
"""

import torch


def _transform(x, temperature, final_logit_softcapping, logit_scale):
    if logit_scale is not None and logit_scale != 1.0:
        x = x * logit_scale
    if final_logit_softcapping:
        x = final_logit_softcapping * torch.tanh(x / final_logit_softcapping)
    if temperature and temperature != 1.0:
        x = x / temperature
    return x


def apply(
    hidden,
    weight,
    bias,
    labels,
    temperature=1.0,
    chunk_size=8192,
    final_logit_softcapping=None,
    logit_scale=None,
    outputs=("log_probs",),
):
    """Chunked per-token log-probs/entropy, differentiable end to end.

    Same signature/semantics as trl.kernels.ChunkedLogProbFunction.apply.
    """
    n = hidden.shape[0]
    vocab = weight.shape[0]
    chunk = max(1024, int(chunk_size or 8192))
    dev = hidden.device
    f32 = torch.float32

    running_max = torch.full((n,), float("-inf"), device=dev, dtype=f32)
    sum_exp = torch.zeros((n,), device=dev, dtype=f32)
    target_logit = torch.zeros((n,), device=dev, dtype=f32)

    max_logit = torch.full((n,), float("-inf"), device=dev, dtype=f32)
    argmax = torch.zeros((n,), device=dev, dtype=torch.long)

    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        z = hidden @ weight[start:end].t()
        if bias is not None:
            z = z + bias[start:end]
        z = _transform(z, temperature, final_logit_softcapping, logit_scale)

        sel = (labels >= start) & (labels < end)
        if sel.any():
            idx = sel.nonzero(as_tuple=True)[0]
            target_logit = target_logit.index_copy(
                0, idx, z[idx, (labels[idx] - start).long()])

        with torch.no_grad():
            cm, ci = z.max(dim=1)
            better = cm > max_logit
            argmax = torch.where(better, ci + start, argmax)
            max_logit = torch.maximum(max_logit, cm)

        new_max = torch.maximum(running_max, cm.detach())
        sum_exp = (sum_exp * torch.exp(running_max - new_max)
                   + torch.exp(z - new_max.unsqueeze(1)).sum(dim=1))
        running_max = new_max

    log_z = running_max + torch.log(sum_exp)
    log_probs = target_logit - log_z
    is_top1 = argmax == labels

    entropy = None
    if "entropy" in outputs or "mean_logits" in outputs:
        entropy = torch.zeros((n,), device=dev, dtype=f32)
        mean_logits = torch.zeros((n,), device=dev, dtype=f32) if "mean_logits" in outputs else None
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            z = hidden @ weight[start:end].t()
            if bias is not None:
                z = z + bias[start:end]
            z = _transform(z, temperature, final_logit_softcapping, logit_scale)
            p = torch.exp(z - log_z.unsqueeze(1))
            entropy = entropy - (p * (z - log_z.unsqueeze(1))).sum(dim=1)
            if mean_logits is not None:
                mean_logits = mean_logits + (p * z).sum(dim=1)
    else:
        mean_logits = None

    return (log_probs,
            entropy if "entropy" in outputs else None,
            None,
            mean_logits if "mean_logits" in outputs else None,
            is_top1 if "is_top1" in outputs else None)


class ChunkedLogProbFunction:
    """Shim matching trl.kernels.ChunkedLogProbFunction's `.apply` usage."""

    @staticmethod
    def apply(hidden, weight, bias, labels, temperature, chunk_size,
              final_logit_softcapping, logit_scale, outputs):
        return apply(hidden, weight, bias, labels, temperature, chunk_size,
                     final_logit_softcapping, logit_scale, tuple(outputs or ("log_probs",)))


def install():
    """Patch TRL's utils module if the Triton kernel is unavailable."""
    import trl.trainer.utils as tu
    if getattr(tu, "_ChunkedLogProbFunction", None) is None:
        tu._ChunkedLogProbFunction = ChunkedLogProbFunction
        return True
    return False
