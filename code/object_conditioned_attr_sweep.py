"""Frozen-checkpoint validation of object-conditioned local attribute evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from experiment import METRICS, build, config_for, pair_tensor, write_json
from model.object_conditioned import (
    hard_neighborhood_weights,
    normalized_map_entropy,
    pair_conditioned_attribute_logits,
    soft_object_weights,
)
import test as evaluation
from utils import set_seed


PROJECT = Path(__file__).resolve().parents[1]


def encode_text_once(model, pairs, config):
    text = [[], None, None]
    for start in range(0, len(pairs), config.text_encoder_batch_size):
        part = model.encode_text_for_open(
            pairs[start:start + config.text_encoder_batch_size])
        text[0].append(part[0])
        if text[1] is None:
            text[1], text[2] = part[1], part[2]
    text[0] = torch.cat(text[0])
    return text


def cmt_text_features(model, text_features, patch_features, batch_size, config):
    outputs = []
    normalized_patches = model.patch_norm(patch_features)
    chunk_size = config.cmt_candidate_batch_size
    for start in range(0, text_features.shape[0], chunk_size):
        original = text_features[start:start + chunk_size]
        transformed = original.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in model.cmt:
            transformed = layer(transformed, normalized_patches)
        transformed = original + model.lamda * transformed
        outputs.append(F.normalize(transformed, dim=-1))
    return torch.cat(outputs, dim=1)


def metrics(dataset, scores, labels, config):
    stats = evaluation.test(
        dataset,
        evaluation.Evaluator(dataset, None),
        torch.cat(scores),
        *[torch.cat(values) for values in labels],
        config,
    )
    return {key: float(stats[key]) for key in METRICS}


def candidate_key(mode, alpha):
    return "{}:{:.2f}".format(mode, alpha)


def flip_permutation(tokens, device):
    grid = int(round(math.sqrt(tokens)))
    if grid * grid != tokens:
        raise ValueError("Patch tokens do not form a square grid")
    columns = torch.arange(grid, device=device)
    return (torch.arange(grid, device=device)[:, None]
            * grid + columns.flip(0)[None, :]).reshape(-1)


def object_similarities(model, images, text, object_layer, config):
    _, final_patches, layers = model.encode_image_with_layers(
        images, [object_layer])
    object_text = cmt_text_features(
        model, text[2], final_patches, images.shape[0], config)
    object_patches = F.normalize(layers[object_layer][:, 1:, :], dim=-1)
    return torch.einsum(
        "bld,bod->blo", object_patches, F.normalize(object_text, dim=-1))


def stability_metrics(original_sim, flipped_sim, object_targets, temperature):
    batch = original_sim.shape[0]
    index = torch.arange(batch, device=original_sim.device)
    original = original_sim[index, :, object_targets]
    flipped = flipped_sim[index, :, object_targets]
    permutation = flip_permutation(original.shape[1], original.device)
    flipped = flipped[:, permutation]

    original_soft = torch.softmax(original / temperature, dim=1)
    flipped_soft = torch.softmax(flipped / temperature, dim=1)
    original_centered = original_soft - original_soft.mean(dim=1, keepdim=True)
    flipped_centered = flipped_soft - flipped_soft.mean(dim=1, keepdim=True)
    correlation = F.cosine_similarity(
        original_centered, flipped_centered, dim=1, eps=1e-8)

    original_center = original.argmax(dim=1)
    flipped_center = flipped.argmax(dim=1)
    original_hard = hard_neighborhood_weights(
        original_sim, kernel=3)[index, :, object_targets]
    flipped_hard = hard_neighborhood_weights(
        flipped_sim, kernel=3)[index, :, object_targets][:, permutation]
    intersection = ((original_hard > 0) & (flipped_hard > 0)).sum(dim=1).float()
    union = ((original_hard > 0) | (flipped_hard > 0)).sum(dim=1).float()

    true_soft = soft_object_weights(
        original_sim, temperature)[index, :, object_targets].unsqueeze(-1)
    entropy = normalized_map_entropy(true_soft).squeeze(-1)
    return {
        "object_top1_consistency": (original_center == flipped_center).float(),
        "object_soft_map_correlation": correlation,
        "object_hard3_iou": intersection / union.clamp_min(1.0),
        "object_soft_entropy": entropy,
    }


def evaluate(model, dataset, config, object_layer, attribute_layer, alphas,
             mask_temperature, batch_size, smoke_batches=0, selected=None,
             measure_stability=True):
    model.eval()
    pairs = pair_tensor(dataset)
    text = encode_text_once(model, pairs, config)
    if selected is None:
        keys = ["cls:0.00"]
        keys.extend(candidate_key(mode, alpha)
                    for mode in ("soft", "hard3") for alpha in alphas)
    else:
        keys = [selected]
    scores = {key: [] for key in keys}
    labels = [[], [], []]
    stability = {}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    layers_requested = sorted(set([object_layer, attribute_layer]))

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch[0].cuda().type(model.clip.dtype)
            cls_features, final_patches, layers = model.encode_image_with_layers(
                images, layers_requested)
            composition_text = cmt_text_features(
                model, text[0], final_patches, images.shape[0], config)
            attribute_text = cmt_text_features(
                model, text[1], final_patches, images.shape[0], config)
            object_text = cmt_text_features(
                model, text[2], final_patches, images.shape[0], config)
            scale = model.clip.logit_scale.exp()

            normalized_cls = F.normalize(cls_features, dim=-1)
            cls_logits = torch.einsum(
                "bd,bkd->bk", normalized_cls, composition_text * scale)
            attr_features = F.normalize(
                model.attr_disentangler(cls_features), dim=-1)
            obj_features = F.normalize(
                model.obj_disentangler(cls_features), dim=-1)
            attr_logits = torch.einsum(
                "bd,bad->ba", attr_features, attribute_text * scale)
            obj_logits = torch.einsum(
                "bd,bod->bo", obj_features, object_text * scale)

            attr_patches = F.normalize(
                layers[attribute_layer][:, 1:, :], dim=-1)
            obj_patches = F.normalize(
                layers[object_layer][:, 1:, :], dim=-1)
            attr_sim = torch.einsum(
                "bld,bad->bla", attr_patches, F.normalize(attribute_text, dim=-1))
            obj_sim = torch.einsum(
                "bld,bod->blo", obj_patches, F.normalize(object_text, dim=-1))
            weights = {
                "soft": soft_object_weights(obj_sim, mask_temperature),
                "hard3": hard_neighborhood_weights(obj_sim, kernel=3),
            }
            local_logits = {
                mode: pair_conditioned_attribute_logits(
                    attr_sim, mode_weights, pairs, scale)
                for mode, mode_weights in weights.items()
            }

            for key in keys:
                if key == "cls:0.00":
                    composition_logits = cls_logits.clone()
                else:
                    mode, alpha_text = key.split(":")
                    composition_logits = (
                        cls_logits + float(alpha_text) * local_logits[mode])
                prediction = [
                    composition_logits,
                    attr_logits.clone(),
                    obj_logits.clone(),
                ]
                scores[key].append(model.logit_infer(prediction, pairs).cpu())

            if measure_stability:
                flipped = torch.flip(images, dims=[3])
                flipped_obj_sim = object_similarities(
                    model, flipped, text, object_layer, config)
                batch_stability = stability_metrics(
                    obj_sim, flipped_obj_sim, batch[2].cuda(), mask_temperature)
                for name, values in batch_stability.items():
                    stability.setdefault(name, []).append(values.cpu())
            for index in range(3):
                labels[index].append(batch[index + 1])

            if smoke_batches and batch_index + 1 >= smoke_batches:
                break

    if smoke_batches:
        return {
            "status": "smoke-completed",
            "batches": smoke_batches,
            "finite": all(torch.isfinite(torch.cat(value)).all().item()
                          for value in scores.values()),
            "score_shapes": {key: list(torch.cat(value).shape)
                             for key, value in scores.items()},
            "stability_samples": int(sum(x.numel()
                                         for x in stability.get(
                                             "object_soft_entropy", []))),
        }

    grid = {key: metrics(dataset, value, labels, config)
            for key, value in scores.items()}
    summarized_stability = {}
    for name, chunks in stability.items():
        values = torch.cat(chunks).float()
        summarized_stability[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
        }
    return grid, summarized_stability


def select_on_validation(grid):
    candidates = []
    for key, stats in grid.items():
        alpha = 0.0 if key == "cls:0.00" else float(key.split(":")[1])
        candidates.append((stats["AUC"], key == "cls:0.00", -alpha, key))
    return max(candidates)[-1]


def run(args):
    output = PROJECT / "diagnostics" / args.run_id
    if output.exists():
        raise FileExistsError("Diagnostic directory already exists: {}".format(output))
    output.mkdir(parents=True)

    base_args = argparse.Namespace(
        dataset=args.dataset,
        seed=args.seed,
        pooling="cls",
        patch_branches="composition",
        exclude_cls=True,
        patch_topk=2,
        epochs=None,
    )
    config = config_for(base_args)
    config.num_workers = 4
    set_seed(config.seed)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT / checkpoint
    expected = json.loads(
        (checkpoint.parent / "result.json").read_text(encoding="utf-8"))

    write_json(output / "provenance.json", {
        "protocol": "fixed-checkpoint/validation-only/object-conditioned-attribute-residual",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "object_layer": args.object_layer,
        "attribute_layer": args.attribute_layer,
        "alphas": args.alphas,
        "mask_temperature": args.mask_temperature,
        "modes": ["soft", "hard3"],
        "test_policy": "do not evaluate test unless both seed validations select non-CLS",
    })
    write_json(output / "config.json", vars(config))

    try:
        train = CompositionDataset(config.dataset_path, "train")
        validation = CompositionDataset(config.dataset_path, "val")
        model = build(config, train)
        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        model.eval()
        result = evaluate(
            model,
            validation,
            config,
            args.object_layer,
            args.attribute_layer,
            args.alphas,
            args.mask_temperature,
            args.batch_size,
            smoke_batches=args.smoke_batches,
        )
        if args.smoke_batches:
            result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
            write_json(output / "smoke.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return

        grid, stability = result
        cls_auc = grid["cls:0.00"]["AUC"]
        if abs(cls_auc - expected["best_val_auc"]) > 1e-6:
            raise AssertionError(
                "CLS validation reproduction failed: {} vs {}".format(
                    cls_auc, expected["best_val_auc"]))
        selected = select_on_validation(grid)
        result = {
            "status": "completed",
            "split": "validation",
            "selection_metric": "validation AUC",
            "selected": selected,
            "selected_val_auc": grid[selected]["AUC"],
            "cls_val_auc": cls_auc,
            "improvement": grid[selected]["AUC"] - cls_auc,
            "grid": grid,
            "object_stability": stability,
            "test_evaluated": False,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        }
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {
            "status": "failed",
            "error": repr(exc),
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="ut-zappos")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object-layer", type=int, default=12)
    parser.add_argument("--attribute-layer", type=int, default=6)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--mask-temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("alphas must be positive")
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run(args)
