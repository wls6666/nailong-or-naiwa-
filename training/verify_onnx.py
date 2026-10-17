"""Three-class PyTorch vs ONNX verification over all real test images by default."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from common import load_model, save_json, sha256
from dataset import ROOT, DEFAULT_DATA, datasets
from inference import softmax, decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', default=ROOT / 'checkpoints/best_model_3class.pth')
    p.add_argument('--onnx', default=ROOT / 'web/model.onnx')
    p.add_argument('--limit', type=int, default=0, help='0 verifies every test image')
    p.add_argument('--tolerance', type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint); ds = datasets(args.data)['test']
    config = json.loads((ROOT / 'web/model_config.json').read_text())
    if ds.class_to_idx != saved['class_to_idx'] or config['class_to_idx'] != ds.class_to_idx: raise RuntimeError('Mapping mismatch')
    if config['checkpoint_sha256'] != sha256(args.checkpoint) or config['model_sha256'] != sha256(args.onnx): raise RuntimeError('Artifact mismatch')
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    # Round-robin indices makes --limit cover classes rather than only the first folder.
    groups = [[i for i, (_, y) in enumerate(ds.samples) if y == k] for k in range(3)]
    order = [group[j] for j in range(max(map(len, groups))) for group in groups if j < len(group)]
    if args.limit: order = order[:args.limit]
    rows = []
    for i in order:
        tensor, label = ds[i]; tensor = tensor.unsqueeze(0)
        with torch.inference_mode(): pt = model(tensor).numpy()
        ox = session.run(['logits'], {'input': tensor.numpy()})[0]
        if pt.shape != (1,3) or ox.shape != (1,3): raise RuntimeError('Model is not three-class')
        pp, op = softmax(pt), softmax(ox)
        pd, _, _ = decisions(pp, ds.class_to_idx, config['reject_threshold'])
        od, _, _ = decisions(op, ds.class_to_idx, config['reject_threshold'])
        row = {'image': str(Path(ds.samples[i][0]).relative_to(ds.root)), 'label': label,
               'pytorch_logits': pt[0].tolist(), 'onnx_logits': ox[0].tolist(),
               'pytorch_probabilities': pp[0].tolist(), 'onnx_probabilities': op[0].tolist(),
               'logit_difference': float(np.max(np.abs(pt-ox))),
               'probability_difference': float(np.max(np.abs(pp-op)))}
        row['pass'] = bool(np.isfinite(pt).all() and np.isfinite(ox).all() and row['logit_difference'] < args.tolerance
                           and row['probability_difference'] < args.tolerance and pt.argmax() == ox.argmax() and pd[0] == od[0])
        rows.append(row)
    report = {'status': 'PASS' if rows and all(r['pass'] for r in rows) else 'FAIL', 'sample_count': len(rows),
              'class_to_idx': ds.class_to_idx, 'tolerance': args.tolerance,
              'max_logit_difference': max(r['logit_difference'] for r in rows),
              'max_probability_difference': max(r['probability_difference'] for r in rows),
              'checkpoint_sha256': sha256(args.checkpoint), 'model_sha256': sha256(args.onnx), 'samples': rows}
    save_json(ROOT / 'results/onnx_verification_3class.json', report)
    print(report['status'],len(rows),'real test images; max logits difference',report['max_logit_difference'],
          'max probability difference',report['max_probability_difference'])
    if report['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
