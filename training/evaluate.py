"""Raw three-class test metrics, plus separately labelled calibrated-policy metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from dataset import DEFAULT_DATA, ROOT, datasets
from common import load_model, save_json, sha256
from inference import infer_dataset, softmax, decisions, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', default=ROOT / 'checkpoints/best_model_3class.pth')
    p.add_argument('--calibrated', action='store_true', help='Apply existing validation-selected policy; never fit on test')
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint)
    ds = datasets(args.data)['test']; mapping = ds.class_to_idx
    if mapping != saved['class_to_idx'] or len(mapping) != 3: raise ValueError('Three-class mapping mismatch')
    labels, logits = infer_dataset(model, ds); probabilities = softmax(logits)
    raw = probabilities.argmax(1)
    rejected = np.zeros(len(ds), dtype=bool)
    predictions = raw
    threshold = None
    if args.calibrated:
        calibration = json.loads((ROOT / 'results/rejection_calibration.json').read_text())
        if calibration['checkpoint_sha256'] != sha256(args.checkpoint): raise RuntimeError('Calibration belongs to another model')
        threshold = calibration['reject_threshold']
        predictions, rejected, _ = decisions(probabilities, mapping, threshold)
    report = {'split': 'test', 'sample_count': len(ds), 'class_to_idx': mapping,
              'policy': 'validation-calibrated rejection' if args.calibrated else 'raw three-class argmax',
              'reject_threshold': threshold, 'checkpoint_epoch': saved['epoch'],
              'checkpoint_sha256': sha256(args.checkpoint), **metrics(labels, predictions, mapping)}
    suffix = '_3class_calibrated' if args.calibrated else '_3class'
    save_json(ROOT / f'results/metrics{suffix}.json', report)
    save_json(ROOT / f'results/test_predictions{suffix}.json', [
        {'image': str(Path(path).relative_to(ds.root)), 'label': int(labels[i]),
         'raw_prediction': int(raw[i]), 'prediction': int(predictions[i]), 'threshold_rejected': bool(rejected[i]),
         'logits': logits[i].tolist(), 'probabilities': probabilities[i].tolist()}
        for i, (path, _) in enumerate(ds.samples)])
    ConfusionMatrixDisplay(np.asarray(report['confusion_matrix']), display_labels=ds.classes).plot(cmap='Greens', colorbar=False)
    plt.title('Test — ' + report['policy']); plt.tight_layout()
    plt.savefig(ROOT / f'results/confusion_matrix{suffix}.png', dpi=160); plt.close()
    binary = json.loads((ROOT / 'results/binary/metrics.json').read_text())
    comparison_path = ROOT / 'results/upgrade_comparison.json'
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {'binary_target_test_accuracy': binary['accuracy']}
    comparison['calibrated' if args.calibrated else 'raw_3class'] = {
        'target_test_accuracy': report['target_accuracy'], 'other_recall': report['per_class']['other']['recall'],
        'other_false_accept_rate': report['other_false_accept_rate'], 'macro_f1': report['macro_f1']}
    save_json(comparison_path, comparison)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
