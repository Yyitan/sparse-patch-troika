"""Controlled Troika pooling experiments with validation-only model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from model.model_factory import get_model
from parameters import parser
import test as evaluation
from utils import load_args, set_seed, get_optimizer, get_scheduler, step_scheduler

PROJECT = Path(__file__).resolve().parents[1]
METRICS = ('best_seen', 'best_unseen', 'best_hm', 'AUC', 'attr_acc', 'obj_acc')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def config_for(args):
    config = parser.parse_args([])
    load_args(str(Path(__file__).resolve().parent / 'config/troika' / (args.dataset + '.yml')), config)
    config.dataset_path = str(PROJECT / 'external/datasets' / args.dataset)
    config.clip_arch = str(PROJECT / 'external/checkpoints/ViT-L-14.pt')
    config.seed = args.seed
    config.visual_pooling = args.pooling
    config.patch_branches = args.patch_branches
    config.patch_include_cls = not args.exclude_cls
    config.patch_topk = args.patch_topk
    config.patch_score_layer = getattr(args, 'patch_score_layer', None)
    config.patch_residual_alpha = getattr(args, 'patch_residual_alpha', 1.0)
    config.patch_lse_temperature = 0.07
    config.normalize_patch_once = True
    config.num_workers = 4
    config.save_final_model = False
    config.open_world = False
    config.epochs = args.epochs or config.epochs
    # Use identical batch size, optimizer and random seed across the three arms.
    config.cmt_candidate_batch_size = 256
    return config


def build(config, dataset):
    return get_model(config,
                     attributes=[x.replace('.', ' ').lower() for x in dataset.attrs],
                     classes=[x.replace('.', ' ').lower() for x in dataset.objs],
                     offset=len(dataset.attrs)).cuda()


def pair_tensor(dataset, training=False):
    pairs = dataset.train_pairs if training else dataset.pairs
    return torch.tensor([(dataset.attr2idx[a], dataset.obj2idx[o])
                         for a, o in pairs], device='cuda')


def evaluate(model, dataset, config):
    model.eval()
    # Encode primitives once; the upstream text-first helper duplicates them
    # when composition texts are split into several chunks.
    pairs = pair_tensor(dataset)
    with torch.no_grad():
        text = [[], None, None]
        for start in range(0, len(pairs), config.text_encoder_batch_size):
            part = model.encode_text_for_open(pairs[start:start + config.text_encoder_batch_size])
            text[0].append(part[0])
            if text[1] is None:
                text[1], text[2] = part[1], part[2]
        text[0] = torch.cat(text[0])
        scores, labels = [], [[], [], []]
        loader = DataLoader(dataset, batch_size=config.eval_batch_size,
                            num_workers=config.num_workers, shuffle=False)
        for batch in loader:
            predictions = model.forward_for_open(batch, text)
            scores.append(model.logit_infer(predictions, pairs).cpu())
            for i in range(3):
                labels[i].append(batch[i + 1])
    stats = evaluation.test(dataset, evaluation.Evaluator(dataset, None),
                            torch.cat(scores), *[torch.cat(x) for x in labels], config)
    return {k: float(stats[k]) for k in METRICS}


def smoke(config):
    dataset = CompositionDataset(config.dataset_path, 'train')
    model = build(config, dataset)
    pairs = pair_tensor(dataset, training=True)
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    model.eval()
    # A learned affine norm exposes repeated-normalization discrepancies.
    with torch.no_grad():
        model.patch_norm.weight.fill_(1.2)
        model.patch_norm.bias.fill_(0.05)
    checks = []
    for mode, include in [('cls', True), ('patch_max', True), ('patch_max', False),
                          ('patch_topk_lse', False)]:
        model.visual_pooling, model.patch_include_cls = mode, include
        model.patch_topk = 2
        with torch.no_grad():
            direct = model(batch, pairs)
            cached = model.forward_for_open(batch, model.encode_text_for_open(pairs))
            error = max(float((a - b).abs().max()) for a, b in zip(direct, cached))
            assert all(torch.allclose(a, b, atol=2e-5, rtol=2e-5)
                       for a, b in zip(direct, cached)), (mode, error)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model.loss_calu(model(batch, pairs), batch)
        assert torch.isfinite(loss), mode
        loss.backward()
        grads = [p.grad for p in model.additional_visual_params.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), mode
        assert sum(float(g.abs().sum()) for g in grads) > 0, mode
        if model.patch_branches == 'composition':
            for head in (model.attr_disentangler, model.obj_disentangler):
                head_grads = [p.grad for p in head.parameters() if p.grad is not None]
                assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
                assert sum(float(g.abs().sum()) for g in head_grads) > 0
        checks.append(dict(mode=mode, include_cls=include, max_error=error, loss=float(loss)))
        model.eval()
    print(json.dumps({'checks': checks, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}, indent=2))


def run(args, config):
    # Full-batch pilot peak is ~33 GiB; recheck before allocating the model.
    while torch.cuda.mem_get_info()[0] < 37000 * 1024 ** 2:
        print('Waiting for at least 37000 MiB free on selected GPU', flush=True)
        time.sleep(30)
    out = PROJECT / 'experiments' / args.run_id
    if out.exists():
        raise FileExistsError('Run directory already exists: {}'.format(out))
    out.mkdir(parents=True)
    config.save_path = str(out)
    write_json(out / 'config.json', vars(config))
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).resolve().parent.rglob('*.py')}
    write_json(out / 'provenance.json', dict(code_sha256=hashes, torch=torch.__version__,
               cuda=torch.version.cuda, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               protocol='corrected-train-mode/multilayer-residual-patch/CMT-retained/v3'))
    try:
        train = CompositionDataset(config.dataset_path, 'train')
        val = CompositionDataset(config.dataset_path, 'val')
        model = build(config, train)
        optimizer = get_optimizer(model, config)
        loader = DataLoader(train, batch_size=config.train_batch_size, shuffle=True,
                            num_workers=config.num_workers)
        scheduler = get_scheduler(optimizer, config, len(loader))
        pairs = pair_tensor(train, True)
        best = -float('inf')
        start_time = time.time()
        history = []
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step, batch in enumerate(loader):
                loss = model.loss_calu(model(batch, pairs), batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss at epoch {} step {}'.format(epoch, step))
                # All pilot configurations use accumulation=1.
                if config.gradient_accumulation_steps != 1:
                    raise ValueError('Pilot requires accumulation=1')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_scheduler(scheduler, config, step, len(loader))
                losses.append(float(loss.detach()))
                if step % 50 == 0:
                    progress = dict(status='training', epoch=epoch + 1, step=step,
                                    steps=len(loader), loss=float(np.mean(losses[-50:])),
                                    elapsed_s=time.time() - start_time)
                    write_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            stats = evaluate(model, val, config)
            record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val=stats,
                          elapsed_s=time.time() - start_time)
            history.append(record)
            write_json(out / 'history.json', history)
            print(json.dumps(record), flush=True)
            if stats['AUC'] > best:
                best, best_epoch = stats['AUC'], epoch + 1
                torch.save(model.state_dict(), str(out / 'val_best.tmp.pt'))
                (out / 'val_best.tmp.pt').replace(out / 'val_best.pt')
        model.load_state_dict(torch.load(str(out / 'val_best.pt'), map_location='cpu'))
        test_data = CompositionDataset(config.dataset_path, 'test')
        stats = evaluate(model, test_data, config)
        result = dict(status='completed', seed=config.seed, dataset=config.dataset,
                      pooling=config.visual_pooling, include_cls=config.patch_include_cls,
                      patch_branches=config.patch_branches, patch_topk=config.patch_topk,
                      patch_score_layer=config.patch_score_layer,
                      patch_residual_alpha=config.patch_residual_alpha,
                      best_epoch=best_epoch, best_val_auc=best, test=stats,
                      elapsed_s=time.time() - start_time,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write_json(out / 'result.json', result)
        write_json(out / 'progress.json', result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(out / 'failure.json', dict(error=repr(exc), status='failed'))
        raise


if __name__ == '__main__':
    options = argparse.ArgumentParser()
    options.add_argument('--run-id', default='smoke')
    options.add_argument('--dataset', default='ut-zappos', choices=['ut-zappos', 'mit-states', 'cgqa'])
    options.add_argument('--pooling', default='cls', choices=['cls', 'patch_max', 'patch_topk_lse'])
    options.add_argument('--exclude-cls', action='store_true')
    options.add_argument('--patch-branches', choices=['all', 'composition'], default='all')
    options.add_argument('--patch-topk', type=int, default=1)
    options.add_argument('--patch-score-layer', type=int)
    options.add_argument('--patch-residual-alpha', type=float, default=1.0)
    options.add_argument('--seed', type=int, default=0)
    options.add_argument('--epochs', type=int)
    options.add_argument('--smoke', action='store_true')
    args = options.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id must be a directory name')
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy('file_system')
    config = config_for(args)
    set_seed(config.seed)
    smoke(config) if args.smoke else run(args, config)
