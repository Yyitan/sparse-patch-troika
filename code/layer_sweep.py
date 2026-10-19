"""Validation sweep of patch evidence from multiple frozen ViT depths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from experiment import METRICS, build, config_for, pair_tensor, write_json
from model.patch_pooling import sparse_patch_logits
import test as evaluation
from utils import set_seed


PROJECT = Path(__file__).resolve().parents[1]
PATCH_MODES = {
    "max": ("patch_max", 1),
    "topk2": ("patch_topk_lse", 2),
}


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
    chunk_size = config.cmt_candidate_batch_size
    normalized_patches = model.patch_norm(patch_features)
    for start in range(0, text_features.shape[0], chunk_size):
        original = text_features[start:start + chunk_size]
        transformed = original.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in model.cmt:
            transformed = layer(transformed, normalized_patches)
        transformed = original + model.lamda * transformed
        outputs.append(transformed / transformed.norm(dim=-1, keepdim=True))
    return torch.cat(outputs, dim=1)


def primitive_logits(model, cls_features, text_features, patch_features, config):
    batch_size = cls_features.shape[0]
    features = [
        model.attr_disentangler(cls_features),
        model.obj_disentangler(cls_features),
    ]
    outputs = []
    for index, branch_features in enumerate(features, start=1):
        branch_features = branch_features / branch_features.norm(
            dim=-1, keepdim=True)
        transformed = cmt_text_features(
            model, text_features[index], patch_features, batch_size, config)
        outputs.append(torch.einsum(
            "bd,bkd->bk",
            branch_features,
            transformed * model.clip.logit_scale.exp(),
        ))
    return outputs


def metrics(dataset, scores, labels, config):
    stats = evaluation.test(
        dataset,
        evaluation.Evaluator(dataset, None),
        torch.cat(scores),
        *[torch.cat(values) for values in labels],
        config,
    )
    return {key: float(stats[key]) for key in METRICS}


def candidate_key(layer, mode, alpha):
    return "layer{:02d}:{}:{:.2f}".format(layer, mode, alpha)


def evaluate_grid(model, dataset, config, layers, modes, alphas,
                  smoke_batches=0, selected=None):
    model.eval()
    pairs = pair_tensor(dataset)
    text = encode_text_once(model, pairs, config)
    if selected is None:
        keys = ["cls:0.00"]
        keys.extend(candidate_key(layer, mode, alpha)
                    for layer in layers for mode in modes for alpha in alphas)
    else:
        keys = [selected]
    scores = {key: [] for key in keys}
    labels = [[], [], []]
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch[0].cuda().type(model.clip.dtype)
            cls_features, final_patches, layer_patches = \
                model.encode_image_with_layers(images, layers)
            normalized_cls = cls_features / cls_features.norm(
                dim=-1, keepdim=True)
            composition_text = cmt_text_features(
                model, text[0], final_patches, images.shape[0], config)
            cls_logits = torch.einsum(
                "bd,bkd->bk",
                normalized_cls,
                composition_text * model.clip.logit_scale.exp(),
            )
            attr_logits, obj_logits = primitive_logits(
                model, cls_features, text, final_patches, config)

            patch_cache = {}
            for key in keys:
                if key == "cls:0.00":
                    composition_logits = cls_logits.clone()
                else:
                    layer_text, mode, alpha_text = key.split(":")
                    layer = int(layer_text.replace("layer", ""))
                    alpha = float(alpha_text)
                    cache_key = (layer, mode)
                    if cache_key not in patch_cache:
                        pooling, topk = PATCH_MODES[mode]
                        patch_cache[cache_key] = sparse_patch_logits(
                            layer_patches[layer],
                            composition_text,
                            model.clip.logit_scale.exp(),
                            mode=pooling,
                            include_cls=False,
                            topk=topk,
                            lse_temperature=config.patch_lse_temperature,
                        )
                    composition_logits = (
                        (1.0 - alpha) * cls_logits
                        + alpha * patch_cache[cache_key]
                    )
                prediction = [
                    composition_logits,
                    attr_logits.clone(),
                    obj_logits.clone(),
                ]
                scores[key].append(model.logit_infer(prediction, pairs).cpu())

            for index in range(3):
                labels[index].append(batch[index + 1])
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break

    if smoke_batches:
        return {
            "batches": smoke_batches,
            "finite": all(torch.isfinite(torch.cat(value)).all().item()
                          for value in scores.values()),
            "shapes": {key: list(torch.cat(value).shape)
                       for key, value in scores.items()},
        }
    return {key: metrics(dataset, value, labels, config)
            for key, value in scores.items()}


def select_on_validation(grid):
    candidates = []
    for key, stats in grid.items():
        alpha = 0.0 if key == "cls:0.00" else float(key.split(":")[-1])
        candidates.append((stats["AUC"], -alpha, key))
    return max(candidates)[-1]


def run(args):
    while torch.cuda.mem_get_info()[0] < args.min_free_mib * 1024 ** 2:
        print("Waiting for at least {} MiB free on selected GPU".format(
            args.min_free_mib), flush=True)
        time.sleep(30)

    output = PROJECT / "experiments" / args.run_id
    if output.exists():
        raise FileExistsError("Run directory already exists: {}".format(output))
    output.mkdir(parents=True)

    base_args = argparse.Namespace(
        dataset=args.dataset,
        seed=args.seed,
        pooling="cls",
        patch_branches="composition",
        exclude_cls=True,
        patch_topk=1,
        epochs=None,
    )
    config = config_for(base_args)
    config.patch_branches = "composition"
    config.patch_include_cls = False
    set_seed(config.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT / checkpoint
    expected = json.loads(
        (checkpoint.parent / "result.json").read_text(encoding="utf-8"))
    p6_grid = json.loads((PROJECT / "experiments" /
                          "P6_ut_p0_fusion_sweep_seed0" /
                          "validation_grid.json").read_text(encoding="utf-8"))["grid"]

    write_json(output / "provenance.json", {
        "protocol": "fixed-P0-checkpoint/validation-layer-sweep/test-selected-once",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "layers": args.layers,
        "modes": args.modes,
        "alphas": args.alphas,
    })
    write_json(output / "config.json", vars(config))

    try:
        train = CompositionDataset(config.dataset_path, "train")
        model = build(config, train)
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state)
        del state

        if args.smoke_batches:
            validation = CompositionDataset(config.dataset_path, "val")
            result = evaluate_grid(
                model, validation, config, args.layers, args.modes,
                args.alphas, smoke_batches=args.smoke_batches)
            write_json(output / "smoke.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return

        validation = CompositionDataset(config.dataset_path, "val")
        grid = evaluate_grid(
            model, validation, config, args.layers, args.modes, args.alphas)
        if abs(grid["cls:0.00"]["AUC"] - expected["best_val_auc"]) > 1e-6:
            raise AssertionError("P0 CLS validation reproduction failed")
        checks = {
            "layer24:max:0.10": p6_grid["max:0.10"]["AUC"],
            "layer24:topk2:0.10": p6_grid["topk2:0.10"]["AUC"],
        }
        for key, expected_auc in checks.items():
            if key in grid and abs(grid[key]["AUC"] - expected_auc) > 1e-6:
                raise AssertionError(
                    "Final-layer reproduction failed for {}: {} vs {}".format(
                        key, grid[key]["AUC"], expected_auc))

        selected = select_on_validation(grid)
        write_json(output / "validation_grid.json", {
            "selected": selected,
            "grid": grid,
            "checks": checks,
        })
        test_data = CompositionDataset(config.dataset_path, "test")
        test_grid = evaluate_grid(
            model,
            test_data,
            config,
            args.layers,
            args.modes,
            args.alphas,
            selected=selected,
        )
        result = {
            "status": "completed",
            "selection_metric": "validation AUC",
            "selected": selected,
            "best_val_auc": grid[selected]["AUC"],
            "p0_val_auc": grid["cls:0.00"]["AUC"],
            "test": test_grid[selected],
            "test_evaluated_keys": [selected],
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
    parser.add_argument("--run-id", default="P7_ut_p0_layer_sweep_seed0")
    parser.add_argument("--dataset", default="ut-zappos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", default="experiments/P0_ut_cls_seed0/val_best.pt")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--modes", nargs="+", choices=sorted(PATCH_MODES),
                        default=["max", "topk2"])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 0.25, 1.0])
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=8000)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    if any(alpha <= 0 or alpha > 1 for alpha in args.alphas):
        raise ValueError("alphas must lie in (0, 1]")
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run(args)
