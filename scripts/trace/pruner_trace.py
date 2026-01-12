"""
pruner_trace.py

Runtime (no-code-change) tracer for `TokenPruner` routing signals.

We avoid editing `prismatic/extern/hf/modeling_prismatic.py` by wrapping methods at runtime.
This keeps Step 5 reversible and low-risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass
class RoutingTrace:
    raw_score: Optional[torch.Tensor] = None  # [B, P, P] (optional; big)
    score_max_per_query: Optional[torch.Tensor] = None  # [B, P]
    score_entropy_per_query: Optional[torch.Tensor] = None  # [B, P]
    indices: Optional[torch.Tensor] = None  # [B, P]
    keep_mask: Optional[torch.Tensor] = None  # [B, P]
    keep_counts: Optional[torch.Tensor] = None  # [B, P]
    pre_seq_len: Optional[int] = None
    post_seq_len: Optional[int] = None
    pre_task_len: Optional[int] = None
    post_task_len: Optional[int] = None


class TokenPrunerTracer:
    """
    Context manager that wraps TokenPruner.forward / get_score / score_to_mask to capture routing signals.
    """

    def __init__(self, pruner: torch.nn.Module, store_raw_score: bool = False) -> None:
        self.pruner = pruner
        self.store_raw_score = store_raw_score

        self.trace = RoutingTrace()

        self._orig_forward: Optional[Callable[..., Any]] = None
        self._orig_get_score: Optional[Callable[..., Any]] = None
        self._orig_score_to_mask: Optional[Callable[..., Any]] = None

    def __enter__(self) -> "TokenPrunerTracer":
        self._orig_forward = getattr(self.pruner, "forward")
        self._orig_get_score = getattr(self.pruner, "get_score")
        self._orig_score_to_mask = getattr(self.pruner, "score_to_mask")

        def wrapped_forward(tokens: torch.Tensor, position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
            try:
                self.trace.pre_seq_len = int(tokens.shape[1])
                num_patches = int(getattr(self.pruner, "num_patches", -1))
                if num_patches > 0:
                    self.trace.pre_task_len = int(tokens.shape[1] - num_patches - 1)
            except Exception:
                pass

            out_tokens, out_position_ids, out_attention_mask = self._orig_forward(tokens, position_ids, attention_mask)

            try:
                self.trace.post_seq_len = int(out_tokens.shape[1])
                if self.trace.keep_mask is not None:
                    num_kept = int(self.trace.keep_mask.sum(dim=-1).to(torch.long)[0].item())
                    self.trace.post_task_len = int(out_tokens.shape[1] - 1 - num_kept)
            except Exception:
                pass

            return out_tokens, out_position_ids, out_attention_mask

        def wrapped_get_score(patches: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
            score = self._orig_get_score(patches, prompts)
            if self.store_raw_score:
                self.trace.raw_score = score.detach()
            try:
                self.trace.score_max_per_query = score.max(dim=-1).values.detach()
            except Exception:
                pass
            try:
                p = torch.softmax(score, dim=-1)
                eps = 1e-12
                self.trace.score_entropy_per_query = (-(p * (p + eps).log()).sum(dim=-1)).detach()
            except Exception:
                pass
            return score

        def wrapped_score_to_mask(score: torch.Tensor) -> torch.Tensor:
            # Mirror TokenPruner.score_to_mask semantics to extract `indices` deterministically.
            indices = score.argmax(dim=-1)
            bsz, num_patches = indices.shape

            keep_counts = []
            for b in range(bsz):
                keep_counts.append(torch.bincount(indices[b], minlength=num_patches))
            keep_counts_t = torch.stack(keep_counts, dim=0)

            mask = self._orig_score_to_mask(score)

            self.trace.indices = indices.detach()
            self.trace.keep_mask = mask.detach()
            self.trace.keep_counts = keep_counts_t.detach()
            return mask

        setattr(self.pruner, "forward", wrapped_forward)
        setattr(self.pruner, "get_score", wrapped_get_score)
        setattr(self.pruner, "score_to_mask", wrapped_score_to_mask)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_forward is not None:
            setattr(self.pruner, "forward", self._orig_forward)
        if self._orig_get_score is not None:
            setattr(self.pruner, "get_score", self._orig_get_score)
        if self._orig_score_to_mask is not None:
            setattr(self.pruner, "score_to_mask", self._orig_score_to_mask)


def to_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().to("cpu")
