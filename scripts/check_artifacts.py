"""Validate delivered artifact identities, selection rule, and recorded measurements."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from dataset import ROOT
from common import sha256


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    history = read('results/training_history.json')
    best = max(history, key=lambda r: (r['val_accuracy'], -r['val_loss']))
    manifest = read('web/model_config.json')
    metrics = read('results/metrics.json')
    verification = read('results/onnx_verification.json')
    browser = read('results/browser_verification.json')
    assert manifest['checkpoint_epoch'] == metrics['checkpoint_epoch'] == best['epoch']
    checkpoint_hash = sha256(ROOT / 'checkpoints/best_model.pth')
    model_hash = sha256(ROOT / 'web/model.onnx')
    assert all(x['checkpoint_sha256'] == checkpoint_hash for x in [manifest, metrics, verification])
    assert all(x['model_sha256'] == model_hash for x in [manifest, verification, browser])
    assert verification['status'] == browser['status'] == 'PASS'
    assert verification['sample_count'] == metrics['sample_count'] == 73
    assert read('results/class_mapping.json') == manifest['class_to_idx'] == metrics['class_to_idx']
    matrix = metrics['confusion_matrix']
    assert sum(map(sum, matrix)) == metrics['sample_count']
    assert abs(sum(matrix[i][i] for i in range(2)) / 73 - metrics['accuracy']) < 1e-12
    vendor = ROOT / 'web/assets/vendor/onnxruntime-web'
    for file, digest in read('web/assets/vendor/onnxruntime-web/version.json')['sha256'].items():
        assert sha256(vendor / file) == digest, file
    print('PASS: checkpoint selection, artifact hashes, metrics, class mapping, ONNX/browser reports, vendored runtime')


if __name__ == '__main__':
    main()
