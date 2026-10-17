"""Train from scratch; select checkpoints only by validation accuracy/loss."""
import argparse
import platform
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DEFAULT_DATA, ROOT, PREPROCESS, datasets
from common import seed_all, save_json
from model import SmallCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_total, correct, total = 0., 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device).float().view(-1, 1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += y.numel()
            loss_total += loss.item() * y.numel()
            correct += ((logits >= 0) == (y >= .5)).sum().item()
    return loss_total / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    ds = datasets(args.data, training=True)
    gen = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=s == 'train',
                            generator=gen, num_workers=0) for s in ('train', 'val')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=.5)
    history, best_key, stale = [], (-1., float('-inf')), 0
    checkpoint = ROOT / 'checkpoints/best_model.pth'
    checkpoint.parent.mkdir(exist_ok=True)
    save_json(ROOT / 'results/class_mapping.json', ds['train'].class_to_idx)
    config = {**vars(args), 'data': str(args.data), 'device': str(device), 'torch': str(torch.__version__),
              'python': platform.python_version(), 'parameters': sum(p.numel() for p in model.parameters()),
              'optimizer': 'AdamW', 'weight_decay': 1e-4, 'dropout': .35,
              'class_to_idx': ds['train'].class_to_idx, 'preprocessing': PREPROCESS}
    save_json(ROOT / 'results/training_config.json', config)
    print('Training:', config, flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders['val'], criterion, device)
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                   val_loss=val_loss, val_accuracy=val_acc, lr=optimizer.param_groups[0]['lr'],
                   seconds=time.perf_counter() - start)
        history.append(row)
        key = (val_acc, -val_loss)
        if key > best_key:
            best_key, stale = key, 0
            torch.save({'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'val_accuracy': val_acc, 'val_loss': val_loss,
                        'class_to_idx': ds['train'].class_to_idx, 'preprocessing': PREPROCESS,
                        'config': config}, checkpoint)
        else:
            stale += 1
        save_json(ROOT / 'results/training_history.json', history)
        print(f'Epoch {epoch:02d}: train loss={train_loss:.5f} acc={train_acc:.4f} | '
              f'val loss={val_loss:.5f} acc={val_acc:.4f} | {row["seconds"]:.1f}s | stale={stale}', flush=True)
        scheduler.step(val_loss)
        if stale >= args.patience:
            print('Early stopping: validation selection did not improve.', flush=True)
            break
    print(f'Best validation accuracy={best_key[0]:.6f}; saved {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
