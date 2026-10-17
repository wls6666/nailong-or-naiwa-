"""Incremental three-class training; reuse the CNN and select by validation macro recall."""
import argparse
from collections import Counter
import platform
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DEFAULT_DATA, ROOT, PREPROCESS, datasets
from common import seed_all, save_json, load_model, sha256
from model import SmallCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_total, weight_total, correct, total = 0., 0., 0, 0
    class_total = torch.zeros(3, dtype=torch.long)
    class_correct = torch.zeros(3, dtype=torch.long)
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            weighted_sum = criterion(logits, y)
            denominator = criterion.weight[y].sum()
            loss = weighted_sum / denominator
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/validation output; refusing to save a broken model")
            if optimizer is not None:
                loss.backward(); optimizer.step()
            prediction = logits.argmax(1)
            matches = prediction.eq(y)
            total += y.numel(); correct += matches.sum().item()
            loss_total += weighted_sum.item(); weight_total += denominator.item()
            for k in range(3):
                class_total[k] += (y == k).sum().item()
                class_correct[k] += (matches & (y == k)).sum().item()
    recalls = (class_correct.double() / class_total.clamp_min(1)).tolist()
    return {'loss': loss_total / weight_total, 'accuracy': correct / total,
            'recall': recalls, 'balanced_accuracy': sum(recalls) / len(recalls)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--initialize', default=ROOT / 'checkpoints/binary/best_model.pth')
    args = p.parse_args()
    torch.set_num_threads(args.threads); seed_all(args.seed)
    ds = datasets(args.data, training=True)
    mapping = ds['train'].class_to_idx
    if set(mapping) != {'nailong', 'naiwa', 'other'}: raise RuntimeError('Run prepare_other_dataset.py before three-class training')
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=s == 'train',
                            generator=generator, num_workers=0) for s in ('train', 'val')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCNN(num_classes=3)
    if not args.from_scratch:
        binary, old = load_model(args.initialize)
        model.features.load_state_dict(binary.features.state_dict())
        with torch.no_grad():
            # The two target rows preserve the original log-odds relation initially.
            for name, old_index in old['class_to_idx'].items():
                sign = 1. if old_index == 1 else -1.
                model.head[3].weight[mapping[name]].copy_(binary.head[3].weight[0] * sign / 2)
                model.head[3].bias[mapping[name]].copy_(binary.head[3].bias[0] * sign / 2)
    model.to(device)
    counts = Counter(ds['train'].targets)
    weights = torch.tensor([len(ds['train']) / (3 * counts[k]) for k in range(3)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=.5)
    history, best_key, stale = [], (-1., float('-inf')), 0
    checkpoint = ROOT / 'checkpoints/best_model_3class.pth'
    checkpoint.parent.mkdir(exist_ok=True)
    save_json(ROOT / 'results/class_mapping.json', mapping)
    config = {**vars(args), 'data': str(args.data), 'initialize': str(args.initialize), 'device': str(device),
              'torch': str(torch.__version__), 'python': platform.python_version(),
              'parameters': sum(p.numel() for p in model.parameters()), 'optimizer': 'AdamW',
              'weight_decay': 1e-4, 'dropout': .35, 'class_to_idx': mapping, 'preprocessing': PREPROCESS,
              'class_counts': {name: counts[idx] for name, idx in mapping.items()},
              'class_weights': weights.cpu().tolist(), 'criterion': 'class-weighted CrossEntropyLoss',
              'checkpoint_selection': 'validation macro recall, then weighted validation loss',
              'initial_checkpoint_sha256': None if args.from_scratch else sha256(args.initialize),
              'other_manifest_sha256': sha256(ROOT / 'data/other_manifest.json'),
              'hard_negative_manifest_sha256': sha256(ROOT / 'results/hard_negative_manifest.json')}
    save_json(ROOT / 'results/training_config_3class.json', config)
    print('Training:', config, flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val = run_epoch(model, loaders['val'], criterion, device)
        row = dict(epoch=epoch, train_loss=train['loss'], train_accuracy=train['accuracy'],
                   val_loss=val['loss'], val_accuracy=val['accuracy'], val_balanced_accuracy=val['balanced_accuracy'],
                   train_per_class_recall={n: train['recall'][i] for n, i in mapping.items()},
                   val_per_class_recall={n: val['recall'][i] for n, i in mapping.items()},
                   lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter() - start)
        history.append(row)
        key = (val['balanced_accuracy'], -val['loss'])
        if key > best_key:
            best_key, stale = key, 0
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'val_accuracy': val['accuracy'], 'val_loss': val['loss'],
                        'val_balanced_accuracy': val['balanced_accuracy'], 'num_classes': 3,
                        'class_to_idx': mapping, 'preprocessing': PREPROCESS, 'config': config}, checkpoint)
        else: stale += 1
        save_json(ROOT / 'results/training_history_3class.json', history)
        print(f'Epoch {epoch:02d}: train loss={train["loss"]:.5f} acc={train["accuracy"]:.4f} | '
              f'val loss={val["loss"]:.5f} acc={val["accuracy"]:.4f} balanced={val["balanced_accuracy"]:.4f} | '
              f'recall={row["val_per_class_recall"]} | {row["seconds"]:.1f}s | stale={stale}', flush=True)
        scheduler.step(val['loss'])
        if stale >= args.patience:
            print('Early stopping: validation selection did not improve.', flush=True); break
    print(f'Best validation macro recall={best_key[0]:.6f}; saved {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
