"""Evaluate the untouched test split using the best validation-selected checkpoint."""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from dataset import DEFAULT_DATA, ROOT, datasets
from common import load_model, save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', default=ROOT / 'checkpoints/best_model.pth')
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint)
    ds = datasets(args.data)['test']
    if ds.class_to_idx != saved['class_to_idx']:
        raise ValueError('Class mapping differs from checkpoint')
    labels, logits = [], []
    with torch.inference_mode():
        for x, y in DataLoader(ds, batch_size=32):
            labels.extend(y.tolist())
            logits.extend(model(x).flatten().tolist())
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities >= .5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = {
        'split': 'test', 'sample_count': len(ds), 'positive_class': ds.classes[1],
        'class_to_idx': ds.class_to_idx, 'checkpoint_epoch': saved['epoch'],
        'checkpoint_sha256': sha256(args.checkpoint),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'confusion_matrix': matrix.tolist(), 'confusion_matrix_axes': 'rows=true, columns=predicted; class index order',
    }
    save_json(ROOT / 'results/metrics.json', metrics)
    save_json(ROOT / 'results/test_predictions.json', [
        {'image': str(__import__('pathlib').Path(ds.samples[i][0]).relative_to(ds.root)),
         'label': int(labels[i]), 'prediction': int(predictions[i]),
         'logit': float(logits[i]), 'positive_probability': float(probabilities[i])}
        for i in range(len(ds))])
    ConfusionMatrixDisplay(matrix, display_labels=ds.classes).plot(cmap='Greens', colorbar=False)
    plt.title('Untouched test split — best validation checkpoint')
    plt.tight_layout()
    plt.savefig(ROOT / 'results/confusion_matrix.png', dpi=160)
    plt.close()
    print(__import__('json').dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
