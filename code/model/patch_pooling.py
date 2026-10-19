import math

import torch
import torch.nn.functional as F


SUPPORTED_PATCH_POOLING = ("patch_max", "patch_topk_lse")


def sparse_patch_logits(
        patch_features,
        text_features,
        logit_scale,
        mode="patch_max",
        include_cls=True,
        topk=1,
        lse_temperature=0.07):
    """Compute class-conditioned logits from visual tokens.

    Args:
        patch_features: Tensor shaped [batch, tokens, dim]. Token zero is CLS.
        text_features: Tensor shaped [batch, classes, dim] or [classes, dim].
        logit_scale: Scalar tensor or number applied after cosine pooling.
    """
    if mode not in SUPPORTED_PATCH_POOLING:
        raise ValueError("Unsupported patch pooling mode: {}".format(mode))
    if patch_features.dim() != 3:
        raise ValueError("patch_features must have shape [batch, tokens, dim]")
    if text_features.dim() == 2:
        text_features = text_features.unsqueeze(0).expand(
            patch_features.shape[0], -1, -1)
    if text_features.dim() != 3:
        raise ValueError("text_features must have shape [batch, classes, dim]")
    if not include_cls:
        patch_features = patch_features[:, 1:, :]
    if patch_features.shape[1] == 0:
        raise ValueError("No visual tokens remain after excluding CLS")

    patch_features = F.normalize(patch_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    similarities = torch.einsum("bld,bkd->blk", patch_features, text_features)

    if mode == "patch_max":
        pooled = similarities.max(dim=1).values
    else:
        if topk < 1:
            raise ValueError("topk must be at least 1")
        if lse_temperature <= 0:
            raise ValueError("lse_temperature must be positive")
        k = min(int(topk), similarities.shape[1])
        top_values = similarities.topk(k, dim=1).values
        pooled = lse_temperature * (
            torch.logsumexp(top_values / lse_temperature, dim=1)
            - math.log(k)
        )

    return pooled * logit_scale
