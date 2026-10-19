"""Evaluate one validation-selected object-conditioned candidate on test once."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from dataset import CompositionDataset
from experiment import build, config_for, write_json
from object_conditioned_attr_sweep import evaluate
from utils import set_seed


PROJECT = Path(__file__).resolve().parents[1]


def run(args):
    output = PROJECT / "diagnostics" / args.run_id
    if output.exists():
        raise FileExistsError("Diagnostic directory already exists: {}".format(output))
    output.mkdir(parents=True)

    validation_result_path = Path(args.validation_result)
    if not validation_result_path.is_absolute():
        validation_result_path = PROJECT / validation_result_path
    validation_result = json.loads(
        validation_result_path.read_text(encoding="utf-8"))
    if validation_result["selected"] != args.selected:
        raise ValueError("Selected candidate does not match validation result")
    if validation_result["improvement"] <= 0:
        raise ValueError("Validation gate did not pass")

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

    write_json(output / "provenance.json", {
        "protocol": "test-once/selected-on-validation/object-conditioned-attribute-residual",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validation_result": str(validation_result_path),
        "validation_result_sha256": hashlib.sha256(
            validation_result_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected": args.selected,
        "object_layer": args.object_layer,
        "attribute_layer": args.attribute_layer,
        "mask_temperature": args.mask_temperature,
    })

    try:
        train = CompositionDataset(config.dataset_path, "train")
        test = CompositionDataset(config.dataset_path, "test")
        model = build(config, train)
        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        alpha = float(args.selected.split(":")[1])
        grid, _ = evaluate(
            model,
            test,
            config,
            args.object_layer,
            args.attribute_layer,
            [alpha],
            args.mask_temperature,
            args.batch_size,
            selected=args.selected,
            measure_stability=False,
        )
        result = {
            "status": "completed",
            "split": "test",
            "selected": args.selected,
            "selected_on_validation_auc": validation_result["selected_val_auc"],
            "cls_validation_auc": validation_result["cls_val_auc"],
            "validation_improvement": validation_result["improvement"],
            "test": grid[args.selected],
            "test_evaluated_keys": [args.selected],
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
    parser.add_argument("--validation-result", required=True)
    parser.add_argument("--selected", default="soft:0.10")
    parser.add_argument("--object-layer", type=int, default=12)
    parser.add_argument("--attribute-layer", type=int, default=6)
    parser.add_argument("--mask-temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    run(args)
