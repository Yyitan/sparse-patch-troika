"""Validation-only sweep of residual patch evidence on a fixed CLS checkpoint."""
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


def infer_scores(model, cls_prediction, patch_prediction, pairs, alpha):
    if patch_prediction is None:
        composition = cls_prediction[0].clone()
    else:
        composition = ((1.0 - alpha) * cls_prediction[0]
                       + alpha * patch_prediction[0])
    prediction = [composition, cls_prediction[1], cls_prediction[2]]
    return model.logit_infer(prediction, pairs).cpu()


def metrics(dataset, scores, labels, config):
    stats = evaluation.test(
        dataset,
        evaluation.Evaluator(dataset, None),
        torch.cat(scores),
        *[torch.cat(values) for values in labels],
        config,
    )
    return {key: float(stats[key]) for key in METRICS}


def evaluate_grid(model, dataset, config, alphas, modes, smoke_batches=0,
                  include_cls_candidate=True):
    model.eval()
    pairs = pair_tensor(dataset)
    text = encode_text_once(model, pairs, config)
    keys = [("cls", 0.0)] if include_cls_candidate else []
    keys.extend((mode, alpha) for mode in modes for alpha in alphas if alpha > 0)
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
            model.visual_pooling = "cls"
            cls_prediction = model.forward_for_open(batch, text)
            if include_cls_candidate:
                scores[("cls", 0.0)].append(
                    infer_scores(model, cls_prediction, None, pairs, 0.0))

            for mode in modes:
                pooling, topk = PATCH_MODES[mode]
                model.visual_pooling = pooling
                model.patch_topk = topk
                patch_prediction = model.forward_for_open(batch, text)
                for alpha in alphas:
                    if alpha > 0:
                        scores[(mode, alpha)].append(infer_scores(
                            model, cls_prediction, patch_prediction, pairs, alpha))

            for index in range(3):
                labels[index].append(batch[index + 1])
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break

    if smoke_batches:
        return {
            "batches": smoke_batches,
            "finite": all(torch.isfinite(torch.cat(value)).all().item()
                          for value in scores.values()),
            "shapes": {"{}:{:.2f}".format(*key): list(torch.cat(value).shape)
                       for key, value in scores.items()},
        }
    return {"{}:{:.2f}".format(*key): metrics(dataset, value, labels, config)
            for key, value in scores.items()}


def select_on_validation(grid):
    candidates = []
    for key, stats in grid.items():
        mode, alpha_text = key.split(":")
        candidates.append((stats["AUC"], -float(alpha_text), mode, key))
    return max(candidates)[-1]


def run(args):
    while torch.cuda.mem_get_info()[0] < 37000 * 1024 ** 2:
        print("Waiting for at least 37000 MiB free on selected GPU", flush=True)
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
    source_result = checkpoint.parent / "result.json"
    expected = json.loads(source_result.read_text(encoding="utf-8"))

    provenance = {
        "protocol": "fixed-P0-checkpoint/validation-only-fusion-selection/test-once",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "alphas": args.alphas,
        "modes": args.modes,
    }
    write_json(output / "provenance.json", provenance)
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
                model, validation, config, args.alphas, args.modes,
                smoke_batches=args.smoke_batches)
            write_json(output / "smoke.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return

        validation = CompositionDataset(config.dataset_path, "val")
        validation_grid = evaluate_grid(
            model, validation, config, args.alphas, args.modes)
        baseline_auc = validation_grid["cls:0.00"]["AUC"]
        if abs(baseline_auc - expected["best_val_auc"]) > 1e-6:
            raise AssertionError(
                "P0 validation reproduction failed: {} vs {}".format(
                    baseline_auc, expected["best_val_auc"]))

        selected_key = select_on_validation(validation_grid)
        selected_mode, selected_alpha = selected_key.split(":")
        selected_alpha = float(selected_alpha)
        write_json(output / "validation_grid.json", {
            "expected_p0_val_auc": expected["best_val_auc"],
            "selected": selected_key,
            "grid": validation_grid,
        })

        test_data = CompositionDataset(config.dataset_path, "test")
        if selected_mode == "cls":
            selected_modes = []
            selected_alphas = [0.0]
        else:
            selected_modes = [selected_mode]
            selected_alphas = [0.0, selected_alpha]
        test_grid = evaluate_grid(
            model,
            test_data,
            config,
            selected_alphas,
            selected_modes,
            include_cls_candidate=(selected_mode == "cls"),
        )
        test_key = "cls:0.00" if selected_mode == "cls" else selected_key
        result = {
            "status": "completed",
            "checkpoint": str(checkpoint),
            "selection_metric": "validation AUC",
            "selected": selected_key,
            "best_val_auc": validation_grid[selected_key]["AUC"],
            "p0_val_auc": baseline_auc,
            "test": test_grid[test_key],
            "test_evaluated_keys": list(test_grid),
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
    parser.add_argument("--run-id", default="P6_ut_p0_fusion_sweep_seed0")
    parser.add_argument("--dataset", default="ut-zappos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", default="experiments/P0_ut_cls_seed0/val_best.pt")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--modes", nargs="+", choices=sorted(PATCH_MODES),
                        default=["max", "topk2"])
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    if not args.alphas or any(alpha < 0 or alpha > 1 for alpha in args.alphas):
        raise ValueError("alphas must lie in [0, 1]")
    if 0.0 not in args.alphas:
        raise ValueError("alphas must include 0 for the CLS reproduction check")
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run(args)
