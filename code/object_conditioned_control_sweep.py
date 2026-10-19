"""Validation-only controls for object-conditioned patch specificity."""
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
    center_gaussian_object_weights,
    normalized_map_entropy,
    pair_conditioned_attribute_logits,
    shuffled_batch_weights,
    soft_object_weights,
    target_object_attribute_logits,
    uniform_object_weights,
    wrong_object_weights,
)
import test as evaluation
from utils import set_seed


PROJECT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = ["uniform", "center", "correct_object", "wrong_object", "shuffled_mask"]


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
    for start in range(0, text_features.shape[0], config.cmt_candidate_batch_size):
        original = text_features[start:start + config.cmt_candidate_batch_size]
        transformed = original.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in model.cmt:
            transformed = layer(transformed, normalized_patches)
        transformed = original + model.lamda * transformed
        outputs.append(F.normalize(transformed, dim=-1))
    return torch.cat(outputs, dim=1)


def dataset_metrics(dataset, scores, labels, config):
    stats = evaluation.test(
        dataset,
        evaluation.Evaluator(dataset, None),
        torch.cat(scores),
        *[torch.cat(values) for values in labels],
        config,
    )
    return {key: float(stats[key]) for key in METRICS}


def target_entropy(weights, object_targets):
    all_entropy = normalized_map_entropy(weights)
    return all_entropy.gather(1, object_targets.long().unsqueeze(1)).squeeze(1)


def evaluate(model, dataset, config, object_layer, attribute_layer, alpha,
             temperature, center_sigma_ratio, batch_size, smoke_batches=0):
    model.eval()
    pairs = pair_tensor(dataset)
    text = encode_text_once(model, pairs, config)
    scores = {"cls": []}
    scores.update({name: [] for name in CONTROL_NAMES})
    local_logits = {name: [] for name in CONTROL_NAMES}
    entropies = {name: [] for name in CONTROL_NAMES}
    labels = [[], [], []]
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
            attribute_targets = batch[1].cuda()
            object_targets = batch[2].cuda()
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

            correct = soft_object_weights(obj_sim, temperature)
            controls = {
                "uniform": uniform_object_weights(obj_sim),
                "center": center_gaussian_object_weights(
                    obj_sim, sigma_ratio=center_sigma_ratio),
                "correct_object": correct,
                "wrong_object": wrong_object_weights(correct),
                "shuffled_mask": shuffled_batch_weights(correct),
            }

            base_prediction = [cls_logits, attr_logits, obj_logits]
            scores["cls"].append(
                model.logit_infer(base_prediction, pairs).cpu())
            for name, weights in controls.items():
                pair_local = pair_conditioned_attribute_logits(
                    attr_sim, weights, pairs, scale)
                prediction = [
                    cls_logits + alpha * pair_local,
                    attr_logits.clone(),
                    obj_logits.clone(),
                ]
                scores[name].append(model.logit_infer(prediction, pairs).cpu())
                local_logits[name].append(target_object_attribute_logits(
                    attr_sim, weights, object_targets, scale).cpu())
                entropies[name].append(
                    target_entropy(weights, object_targets).cpu())

            for index in range(3):
                labels[index].append(batch[index + 1])
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break

    if smoke_batches:
        return {
            "status": "smoke-completed",
            "batches": smoke_batches,
            "finite": all(
                torch.isfinite(torch.cat(chunks)).all().item()
                for chunks in list(scores.values()) + list(local_logits.values())),
            "score_shapes": {
                key: list(torch.cat(chunks).shape) for key, chunks in scores.items()},
            "local_shapes": {
                key: list(torch.cat(chunks).shape)
                for key, chunks in local_logits.items()},
        }

    grid = {
        name: dataset_metrics(dataset, chunks, labels, config)
        for name, chunks in scores.items()
    }
    attribute_targets = torch.cat(labels[0]).long()
    local_attribute = {}
    for name in CONTROL_NAMES:
        logits = torch.cat(local_logits[name]).float()
        index = torch.arange(logits.shape[0])
        target_scores = logits[index, attribute_targets]
        local_attribute[name] = {
            "accuracy": float((logits.argmax(dim=1) == attribute_targets).float().mean()),
            "target_logit_mean": float(target_scores.mean()),
            "normalized_entropy": float(torch.cat(entropies[name]).float().mean()),
        }

    correct_auc = grid["correct_object"]["AUC"]
    correct_attr = local_attribute["correct_object"]["accuracy"]
    competitors = ["uniform", "center", "wrong_object", "shuffled_mask"]
    gate = {
        "correct_beats_all_pair_auc": all(
            correct_auc > grid[name]["AUC"] for name in competitors),
        "correct_beats_all_local_attribute": all(
            correct_attr > local_attribute[name]["accuracy"]
            for name in competitors),
        "pair_auc_margins": {
            name: correct_auc - grid[name]["AUC"] for name in competitors},
        "local_attribute_margins": {
            name: correct_attr - local_attribute[name]["accuracy"]
            for name in competitors},
    }
    gate["passed"] = (
        gate["correct_beats_all_pair_auc"]
        and gate["correct_beats_all_local_attribute"])
    return {"grid": grid, "local_attribute": local_attribute, "gate": gate}


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
        "protocol": "fixed-checkpoint/validation-only/object-mask-specificity-controls/v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            (Path(__file__).parent / "model/object_conditioned.py").read_bytes()).hexdigest(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "object_layer": args.object_layer,
        "attribute_layer": args.attribute_layer,
        "alpha": args.alpha,
        "mask_temperature": args.mask_temperature,
        "center_sigma_ratio": args.center_sigma_ratio,
        "controls": CONTROL_NAMES,
        "wrong_object_rule": "deterministic object-index roll by one",
        "shuffled_mask_rule": "batch roll by one",
        "test_policy": "validation only; no test evaluation",
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
            args.alpha,
            args.mask_temperature,
            args.center_sigma_ratio,
            args.batch_size,
            smoke_batches=args.smoke_batches,
        )
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        if args.smoke_batches:
            write_json(output / "smoke.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return

        cls_auc = result["grid"]["cls"]["AUC"]
        if abs(cls_auc - expected["best_val_auc"]) > 1e-6:
            raise AssertionError(
                "CLS validation reproduction failed: {} vs {}".format(
                    cls_auc, expected["best_val_auc"]))
        result.update({
            "status": "completed",
            "split": "validation",
            "cls_val_auc": cls_auc,
            "test_evaluated": False,
        })
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
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mask-temperature", type=float, default=0.07)
    parser.add_argument("--center-sigma-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    if args.alpha <= 0:
        raise ValueError("alpha must be positive")
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run(args)
