import math

import torch
import torch.nn.functional as F


def _grid_size(tokens):
    grid = int(round(math.sqrt(tokens)))
    if grid * grid != tokens:
        raise ValueError("Patch tokens do not form a square grid")
    return grid


def soft_object_weights(object_similarities, temperature=0.07):
    """Normalize object-to-patch similarities over spatial tokens."""
    if object_similarities.dim() != 3:
        raise ValueError("object_similarities must have shape [batch, tokens, objects]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(object_similarities / temperature, dim=1)


def uniform_object_weights(object_similarities):
    """Return an object-independent uniform distribution over patches."""
    if object_similarities.dim() != 3:
        raise ValueError("object_similarities must have shape [batch, tokens, objects]")
    tokens = object_similarities.shape[1]
    return torch.full_like(object_similarities, 1.0 / tokens)


def center_gaussian_object_weights(object_similarities, sigma_ratio=0.25):
    """Return a fixed object-independent Gaussian center prior."""
    if object_similarities.dim() != 3:
        raise ValueError("object_similarities must have shape [batch, tokens, objects]")
    if sigma_ratio <= 0:
        raise ValueError("sigma_ratio must be positive")
    batch, tokens, objects = object_similarities.shape
    grid = _grid_size(tokens)
    coordinates = torch.arange(
        grid, device=object_similarities.device, dtype=object_similarities.dtype)
    coordinates = coordinates - (grid - 1.0) / 2.0
    squared_distance = (
        coordinates[:, None].square() + coordinates[None, :].square())
    sigma = grid * sigma_ratio
    weights = torch.exp(-0.5 * squared_distance / (sigma * sigma))
    weights = weights.reshape(1, tokens, 1)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights.expand(batch, tokens, objects)


def wrong_object_weights(object_weights, shift=1):
    """Assign each object the mask of a different deterministic object."""
    if object_weights.dim() != 3:
        raise ValueError("object_weights must have shape [batch, tokens, objects]")
    objects = object_weights.shape[2]
    if objects < 2:
        raise ValueError("at least two objects are required")
    if shift % objects == 0:
        raise ValueError("shift must produce a derangement")
    return object_weights.roll(shifts=-shift, dims=2)


def shuffled_batch_weights(object_weights):
    """Use the same object mask from a different image in the batch."""
    if object_weights.dim() != 3:
        raise ValueError("object_weights must have shape [batch, tokens, objects]")
    if object_weights.shape[0] < 2:
        return object_weights
    return object_weights.roll(shifts=1, dims=0)


def hard_neighborhood_weights(object_similarities, kernel=3):
    """Uniform weights in a square neighborhood around each object's argmax."""
    if object_similarities.dim() != 3:
        raise ValueError("object_similarities must have shape [batch, tokens, objects]")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    batch, tokens, objects = object_similarities.shape
    grid = _grid_size(tokens)
    centers = object_similarities.argmax(dim=1)
    one_hot = F.one_hot(centers, num_classes=tokens).to(object_similarities.dtype)
    maps = one_hot.reshape(batch * objects, 1, grid, grid)
    maps = F.max_pool2d(maps, kernel_size=kernel, stride=1, padding=kernel // 2)
    maps = maps.reshape(batch, objects, tokens)
    maps = maps / maps.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return maps.permute(0, 2, 1).contiguous()


def pair_conditioned_attribute_logits(
        attribute_similarities, object_weights, pairs, logit_scale):
    """Pool attribute evidence using the candidate pair's object region."""
    if attribute_similarities.dim() != 3 or object_weights.dim() != 3:
        raise ValueError("similarities and weights must be rank-three tensors")
    if attribute_similarities.shape[:2] != object_weights.shape[:2]:
        raise ValueError("attribute similarities and object weights must share batch/tokens")
    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [pairs, 2]")
    attr_index = pairs[:, 0].long()
    obj_index = pairs[:, 1].long()
    pair_attributes = attribute_similarities.index_select(2, attr_index)
    pair_weights = object_weights.index_select(2, obj_index)
    return (pair_attributes * pair_weights).sum(dim=1) * logit_scale


def target_object_attribute_logits(
        attribute_similarities, object_weights, object_targets, logit_scale):
    """Pool every attribute using each image's diagnostic target-object mask."""
    if attribute_similarities.dim() != 3 or object_weights.dim() != 3:
        raise ValueError("similarities and weights must be rank-three tensors")
    if attribute_similarities.shape[:2] != object_weights.shape[:2]:
        raise ValueError("attribute similarities and object weights must share batch/tokens")
    if object_targets.dim() != 1 or object_targets.shape[0] != object_weights.shape[0]:
        raise ValueError("object_targets must have shape [batch]")
    batch, tokens, _ = object_weights.shape
    gather_index = object_targets.long().reshape(batch, 1, 1).expand(batch, tokens, 1)
    selected_weights = object_weights.gather(2, gather_index).squeeze(2)
    return torch.einsum(
        "bla,bl->ba", attribute_similarities, selected_weights) * logit_scale


def normalized_map_entropy(weights):
    """Return per-object entropy normalized to [0, 1]."""
    tokens = weights.shape[1]
    probabilities = weights.clamp_min(1e-12)
    return -(probabilities * probabilities.log()).sum(dim=1) / math.log(tokens)
