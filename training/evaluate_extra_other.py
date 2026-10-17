"""Extra real distractors excluded from train/val/test manifests; no web scraping.
These are unused samples of the same source datasets, not an unseen-domain benchmark.
"""
import json
from pathlib import Path
import random
import torch
from dataset import ROOT, load_rgb, resize_rgb
from common import load_model, save_json, sha256
from inference import softmax, decisions


def main():
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / 'data/other_manifest.json').read_text())
    calibration = json.loads((ROOT / 'results/rejection_calibration.json').read_text())
    model, saved = load_model()
    if calibration['checkpoint_sha256'] != sha256(ROOT / 'checkpoints/best_model_3class.pth'): raise RuntimeError('Stale calibration')
    used = {(r['source'], r['source_id']) for r in manifest['samples']}
    cache = ROOT / 'data/source_cache'
    groups = {'cat': [], 'dog': []}
    for line in (cache / 'oxford-iiit-pet/annotations/test.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split(); name, species = parts[0], int(parts[2])
        if ('oxford_pets', name) not in used:
            groups['cat' if species == 1 else 'dog'].append((cache / f'oxford-iiit-pet/images/{name}.jpg', name, 'oxford_pets'))
    for category, classes in {'vehicle': ['pickup_truck', 'bus'], 'food': ['apple', 'orange'],
                              'building': ['house', 'skyscraper'], 'person': ['man', 'woman'],
                              'nature': ['forest', 'sea']}.items():
        groups[category] = []
        for name in classes:
            for path in sorted((cache / 'cifar100/test').glob('*/' + name + '/*')):
                identity = str(path.relative_to(cache / 'cifar100'))
                if ('cifar100', identity) not in used: groups[category].append((path, identity, 'cifar100'))
    records = []
    rng = random.Random(42)
    for category, candidates in groups.items():
        rng.shuffle(candidates)
        if len(candidates) < 2: raise RuntimeError(f'Insufficient unused source images: {category}')
        for path, identity, source in candidates[:2]:
            with torch.inference_mode(): logits = model(resize_rgb(load_rgb(path)).unsqueeze(0)).numpy()
            probabilities = softmax(logits)
            prediction, rejected, maximum = decisions(probabilities, saved['class_to_idx'], calibration['reject_threshold'])
            predicted_class = next(k for k, v in saved['class_to_idx'].items() if v == prediction[0])
            records.append({'domain': category, 'source': source, 'source_id': identity,
                            'path': str(path.relative_to(ROOT)), 'probabilities': probabilities[0].tolist(),
                            'prediction': predicted_class, 'threshold_rejected': bool(rejected[0]),
                            'highest_target_probability': float(maximum[0]), 'actual_other': True})
    report = {'sample_count': len(records), 'checkpoint_sha256': sha256(ROOT / 'checkpoints/best_model_3class.pth'),
              'note': 'Unused official test-pool source samples, disjoint from generated train/val/test. Same-source qualitative probe, not a new-domain guarantee.',
              'false_accept_count': sum(r['prediction'] != 'other' for r in records), 'samples': records}
    save_json(ROOT / 'results/extra_other_probes.json', report)
    for row in records: print(row['domain'],row['source_id'],row['prediction'],row['probabilities'])
    print('Extra Other false accepts:',report['false_accept_count'],'/',len(records))


if __name__ == '__main__': main()
