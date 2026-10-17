"""Verify active three-class artifacts, baseline preservation, and validation-only policy."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from dataset import ROOT
from common import sha256, load_model


def read(name): return json.loads((ROOT / name).read_text())


def main():
    history = read('results/training_history_3class.json')
    best = max(history, key=lambda r: (r['val_balanced_accuracy'], -r['val_loss']))
    manifest = read('web/model_config.json')
    metrics = read('results/metrics_3class.json')
    calibrated = read('results/metrics_3class_calibrated.json')
    verification = read('results/onnx_verification_3class.json')
    browser = read('results/browser_verification_3class.json')
    calibration = read('results/rejection_calibration.json')
    assert manifest['checkpoint_epoch'] == metrics['checkpoint_epoch'] == best['epoch']
    checkpoint_hash = sha256(ROOT / 'checkpoints/best_model_3class.pth')
    model_hash = sha256(ROOT / 'web/model.onnx')
    assert all(x['checkpoint_sha256'] == checkpoint_hash for x in [manifest, metrics, calibrated, verification, calibration])
    assert all(x['model_sha256'] == model_hash for x in [manifest, verification, browser])
    assert verification['status'] == browser['status'] == 'PASS'
    assert read('results/class_mapping.json') == manifest['class_to_idx'] == metrics['class_to_idx'] == calibration['class_to_idx']
    assert len(manifest['classes']) == 3 and manifest['output_shape'] == [1,3]
    audit = read('results/dataset_audit_3class.json')
    expected_val = sum(audit['splits']['val'].values())
    expected_test = sum(audit['splits']['test'].values())
    assert calibration['split'] == 'val' and calibration['sample_count'] == expected_val
    assert manifest['reject_threshold'] == calibration['selected']['threshold']
    assert manifest['rejection']['calibration_sha256'] == sha256(ROOT / 'results/rejection_calibration.json')
    assert read('results/training_config_3class.json')['other_manifest_sha256'] == sha256(ROOT / 'data/other_manifest.json')
    assert read('results/training_config_3class.json')['hard_negative_manifest_sha256'] == sha256(ROOT / 'results/hard_negative_manifest.json')
    for report in [metrics, calibrated]:
        matrix = report['confusion_matrix']; total = report['sample_count']
        assert sum(map(sum,matrix)) == total == verification['sample_count'] == expected_test
        assert abs(sum(matrix[i][i] for i in range(3))/total-report['accuracy']) < 1e-12
        other = manifest['class_to_idx']['other']
        count = sum(matrix[other])-matrix[other][other]
        assert count == report['other_false_accept_count']
        assert abs(count/sum(matrix[other])-report['other_false_accept_rate']) < 1e-12
    binary = read('results/binary/metrics.json')
    assert sha256(ROOT / 'checkpoints/binary/best_model.pth') == binary['checkpoint_sha256']
    assert binary['accuracy'] == 68/73
    legacy, _ = load_model(ROOT / 'checkpoints/binary/best_model.pth')
    active, _ = load_model()
    assert legacy.head[3].out_features == 1 and active.head[3].out_features == 3
    vendor = ROOT / 'web/assets/vendor/onnxruntime-web'
    for file, digest in read('web/assets/vendor/onnxruntime-web/version.json')['sha256'].items():
        assert sha256(vendor / file) == digest, file
    print('PASS: three-class artifacts, validation policy, checkpoint selection, original baseline, metrics and runtime')


if __name__ == '__main__': main()
