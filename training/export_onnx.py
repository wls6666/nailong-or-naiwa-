"""Export the genuinely trained checkpoint; check graph and run a real test image."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from common import load_model, save_json, sha256
from dataset import ROOT, DEFAULT_DATA, discover_splits, load_rgb, resize_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/best_model_3class.pth')
    p.add_argument('--output', type=Path, default=ROOT / 'web/model.onnx')
    p.add_argument('--data', default=DEFAULT_DATA)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model(args.checkpoint)
    calibration = json.loads((ROOT / 'results/rejection_calibration.json').read_text())
    if calibration['checkpoint_sha256'] != sha256(args.checkpoint):
        raise RuntimeError('Run validation calibration for this checkpoint before exporting')
    if len(saved['class_to_idx']) != 3 or model.head[3].out_features != 3:
        raise RuntimeError('Expected a genuinely three-class checkpoint')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), str(args.output),
                      input_names=['input'], output_names=['logits'], opset_version=17,
                      dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
                      dynamo=False)
    graph = onnx.load(str(args.output))
    for key, value in {'class_to_idx': saved['class_to_idx'], 'preprocessing': saved['preprocessing']}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].name == 'input' and session.get_outputs()[0].name == 'logits'
    image = next(discover_splits(args.data)['test'].rglob('*.jpg'))
    tensor = resize_rgb(load_rgb(image)).unsqueeze(0).numpy()
    output = session.run(['logits'], {'input': tensor})[0]
    if output.shape != (1, 3) or not np.isfinite(output).all():
        raise RuntimeError(f'Invalid output: {output}')
    manifest = {'model': args.output.name, 'input_name': 'input', 'output_name': 'logits',
                'input_shape': [1, 3, 128, 128], 'output_shape': [1, 3],
                'task': 'three-class-with-validation-calibrated-rejection',
                'classes': [name for name, idx in sorted(saved['class_to_idx'].items(), key=lambda kv: kv[1])],
                'reject_threshold': calibration['reject_threshold'],
                'rejection': {'other_class': 'other', 'target_classes': ['nailong', 'naiwa'],
                              'calibration_split': 'val', 'calibration_sha256': sha256(ROOT / 'results/rejection_calibration.json'),
                              'low_confidence_display': 'target maximum probability, not Other probability'}, 'class_to_idx': saved['class_to_idx'],
                'preprocessing': saved['preprocessing'], 'checkpoint_epoch': saved['epoch'],
                'checkpoint_sha256': sha256(args.checkpoint), 'model_sha256': sha256(args.output)}
    save_json(args.output.parent / 'model_config.json', manifest)
    report = {'onnx_checker': 'PASS', 'python_runtime': 'PASS', 'opset': 17,
              'bytes': args.output.stat().st_size, 'sample_logits': output[0].tolist(), **manifest}
    save_json(ROOT / 'results/onnx_export_3class.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
