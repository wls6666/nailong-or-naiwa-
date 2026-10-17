"""Compare PyTorch and ONNX over real test images and exactly the same tensors."""
import argparse
import math
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from common import load_model, save_json, sha256
from dataset import ROOT, DEFAULT_DATA, datasets


def sigmoid(x):
    return 1 / (1 + math.exp(-float(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--checkpoint', default=ROOT / 'checkpoints/best_model.pth')
    p.add_argument('--onnx', default=ROOT / 'web/model.onnx')
    p.add_argument('--limit', type=int, default=0, help='0 verifies the entire test split')
    p.add_argument('--tolerance', type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint)
    ds = datasets(args.data)['test']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    if ds.class_to_idx != saved['class_to_idx']:
        raise RuntimeError('Mapping mismatch')
    rows = []
    for i in range(min(args.limit or len(ds), len(ds))):
        tensor, label = ds[i]
        tensor = tensor.unsqueeze(0)
        with torch.inference_mode():
            pt = model(tensor).item()
        ox = float(session.run(['logits'], {'input': tensor.numpy()})[0][0, 0])
        row = {'image': str(Path(ds.samples[i][0]).relative_to(ds.root)), 'label': label,
               'pytorch_logit': pt, 'onnx_logit': ox,
               'pytorch_probability': sigmoid(pt), 'onnx_probability': sigmoid(ox),
               'logit_difference': abs(pt - ox), 'probability_difference': abs(sigmoid(pt) - sigmoid(ox))}
        row['pass'] = bool(np.isfinite([pt, ox]).all() and row['logit_difference'] < args.tolerance
                           and row['probability_difference'] < args.tolerance
                           and (pt >= 0) == (ox >= 0))
        rows.append(row)
    report = {'status': 'PASS' if all(r['pass'] for r in rows) and rows else 'FAIL',
              'sample_count': len(rows), 'tolerance': args.tolerance,
              'max_logit_difference': max(r['logit_difference'] for r in rows),
              'max_probability_difference': max(r['probability_difference'] for r in rows),
              'checkpoint_sha256': sha256(args.checkpoint), 'model_sha256': sha256(args.onnx),
              'samples': rows}
    save_json(ROOT / 'results/onnx_verification.json', report)
    print('PyTorch probability:', rows[0]['pytorch_probability'])
    print('ONNX probability:', rows[0]['onnx_probability'])
    print('Max logit difference:', report['max_logit_difference'])
    print('Max probability difference:', report['max_probability_difference'])
    print(report['status'], len(rows), 'real test images')
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
