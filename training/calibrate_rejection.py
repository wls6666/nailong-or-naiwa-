"""Choose threshold using ONLY validation. Test data never enters the scan."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from dataset import ROOT, DEFAULT_DATA, discover_splits, CharacterDataset
from common import load_model, save_json, sha256
from inference import infer_dataset, softmax, decisions, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/best_model_3class.pth')
    p.add_argument('--max-target-drop', type=float, default=.03)
    p.add_argument('--max-class-drop', type=float, default=.05)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint)
    # Instantiate validation only; no access to test images/predictions in calibration.
    ds = CharacterDataset(discover_splits(args.data)['val'])
    mapping = ds.class_to_idx
    if saved['class_to_idx'] != mapping or len(mapping) != 3: raise RuntimeError('Mapping mismatch')
    labels, logits = infer_dataset(model, ds); probabilities = softmax(logits)
    baseline = metrics(labels, probabilities.argmax(1), mapping)
    candidates = []
    for threshold in np.round(np.arange(.5, .951, .05), 2):
        prediction, rejected, _ = decisions(probabilities, mapping, float(threshold))
        report = metrics(labels, prediction, mapping)
        class_drop = max(baseline['per_class'][k]['recall'] - report['per_class'][k]['recall'] for k in ['nailong', 'naiwa'])
        drop = baseline['target_accuracy'] - report['target_accuracy']
        candidates.append({'threshold': float(threshold), 'target_accuracy_drop': drop,
                           'max_target_class_recall_drop': class_drop, 'rejected_count': int(rejected.sum()),
                           'eligible': drop <= args.max_target_drop + 1e-12 and class_drop <= args.max_class_drop + 1e-12, **report})
    eligible = [r for r in candidates if r['eligible']]
    if eligible:
        selected = min(eligible, key=lambda r: (r['other_false_accept_rate'], -r['macro_f1'], r['threshold']))
    else:
        selected = min(candidates, key=lambda r: (r['target_accuracy_drop'], r['max_target_class_recall_drop'], r['other_false_accept_rate'], -r['macro_f1'], r['threshold']))
    report = {'split': 'val', 'sample_count': len(ds), 'class_to_idx': mapping,
              'checkpoint_sha256': sha256(args.checkpoint), 'reject_threshold': selected['threshold'],
              'selection_rule': 'Among candidates with target accuracy drop <= 3pp and each target recall drop <= 5pp: minimize Other FAR; tie maximize macro F1; tie choose lower threshold. If none eligible, minimize target loss first.',
              'max_target_accuracy_drop': args.max_target_drop, 'max_target_class_recall_drop': args.max_class_drop,
              'constraints_met': bool(eligible), 'baseline': baseline, 'selected': selected, 'candidates': candidates}
    save_json(ROOT / 'results/rejection_calibration.json', report)
    save_json(ROOT / 'results/validation_predictions_3class.json', [
        {'image': str(Path(ds.samples[i][0]).relative_to(ds.root)), 'label': int(labels[i]),
         'logits': logits[i].tolist(), 'probabilities': probabilities[i].tolist()} for i in range(len(ds))])
    print('Validation-only calibration:', json.dumps({k: v for k, v in report.items() if k not in ['candidates', 'baseline', 'selected']}, indent=2))
    for row in candidates: print(row['threshold'], 'targets', round(row['target_accuracy'],4), 'Other FAR', round(row['other_false_accept_rate'],4), 'macro F1', round(row['macro_f1'],4), 'eligible', row['eligible'])


if __name__ == '__main__': main()
