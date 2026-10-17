"""Real Chromium UI inference + Python comparison. Development dependency: playwright.
The HTTP server serves files only; no prediction endpoint exists.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import numpy as np
import onnxruntime as ort
import torch
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from dataset import ROOT, DEFAULT_DATA, datasets, load_rgb, resize_rgb, discover_splits
from common import load_model, save_json, sha256
from inference import softmax, decisions


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    if json.loads((ROOT / 'web/model_config.json').read_text()).get('task') == 'open-set-ood-v3':
        from verify_ood_browser import main as verify_ood
        return verify_ood()
    parser = argparse.ArgumentParser()
    parser.add_argument('--chromium', default='/usr/bin/chromium', help='Use empty string for Playwright-managed Chromium')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, saved = load_model()
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(ROOT / 'web/model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    ds = datasets(DEFAULT_DATA)['test']
    chosen = []
    config = json.loads((ROOT / 'web/model_config.json').read_text())
    for label in [saved['class_to_idx'][name] for name in ['nailong','naiwa']]:
        chosen.extend([Path(path) for path, y in ds.samples if y == label][:2])
    for source_name in ['cifar100','oxford_pets']:
        chosen.append(next(Path(path) for path, y in ds.samples if y == saved['class_to_idx']['other'] and Path(path).name.startswith(source_name)))
    validation = json.loads((ROOT / 'results/validation_predictions_3class.json').read_text())
    for row in validation:
        probabilities = np.asarray([row['probabilities']])
        _, rejected, _ = decisions(probabilities, saved['class_to_idx'], config['reject_threshold'])
        if rejected[0]:
            chosen.append(discover_splits(DEFAULT_DATA)['val'] / row['image']); break
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT / 'web')))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    records, requests, page_errors = [], [], []
    try:
        with tempfile.TemporaryDirectory() as temp, sync_playwright() as playwright:
            kwargs = {'headless': True, 'args': ['--no-sandbox', '--disable-dev-shm-usage']}
            if args.chromium: kwargs['executable_path'] = args.chromium
            browser = playwright.chromium.launch(**kwargs)
            page = browser.new_page(viewport={'width': 1120, 'height': 1050})
            page.on('request', lambda req: requests.append({'url': req.url, 'method': req.method}))
            page.on('pageerror', lambda error: page_errors.append(str(error)))
            page.goto(url)
            page.wait_for_function("document.querySelector('#model-status').dataset.state === 'ready'", timeout=60000)
            page.evaluate("document.addEventListener('inference-complete', event => { window.lastInference = event.detail; });")
            for source in chosen:
                # A lossless PNG of real test pixels isolates preprocessing from decoder differences.
                png = Path(temp) / (source.stem + '.png'); load_rgb(source).save(png)
                for image, kind in [(png, 'lossless-test-pixels'), (source, 'original-jpeg')]:
                    page.evaluate('window.lastInference = null')
                    page.locator('#file-input').set_input_files(str(image))
                    page.wait_for_function("!document.querySelector('#predict-button').disabled")
                    page.locator('#predict-button').click()
                    page.wait_for_function('window.lastInference !== null')
                    actual = page.evaluate('window.lastInference')
                    # Decode a copy the same way as the UI, then retrieve its actual browser tensor.
                    tensor_list = page.evaluate('''async () => {
                        const { imageToCHW } = await import('./preprocess.js');
                        const blob = await (await fetch(document.querySelector('#preview').src)).blob();
                        const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
                        const tensor = Array.from(imageToCHW(bitmap)); bitmap.close(); return tensor;
                    }''')
                    browser_tensor = np.asarray(tensor_list, dtype=np.float32).reshape(1, 3, 128, 128)
                    python_tensor = resize_rgb(load_rgb(image)).unsqueeze(0).numpy()
                    python_logits = session.run(['logits'], {'input': browser_tensor})[0]
                    with torch.inference_mode():
                        pytorch_logits = model(torch.from_numpy(browser_tensor)).numpy()
                    reference_logits = session.run(['logits'], {'input': python_tensor})[0]
                    probabilities = softmax(python_logits)
                    prediction, rejected, highest_target = decisions(probabilities, saved['class_to_idx'], config['reject_threshold'])
                    category = config['classes'][int(prediction[0])]
                    raw_index = int(probabilities[0].argmax())
                    confidence = float(highest_target[0] if rejected[0] else probabilities[0, raw_index])
                    tensor_difference = float(np.abs(browser_tensor - python_tensor).max())
                    actual_logits = np.asarray(actual['logits'])
                    row = {'image': str(source.relative_to(ROOT / 'data/NaiwaOrNailong-dataset/data')), 'kind': kind,
                           'browser': actual, 'python_onnx_same_tensor_logits': python_logits[0].tolist(),
                           'pytorch_same_tensor_logits': pytorch_logits[0].tolist(),
                           'python_preprocessing_logits': reference_logits[0].tolist(),
                           'max_preprocessing_tensor_difference': tensor_difference,
                           'browser_python_logit_difference': float(np.max(np.abs(actual_logits - python_logits[0]))),
                           'displayed_confidence': page.locator('#confidence-value').inner_text()}
                    assert actual['outputShape'] == [1,3]
                    assert actual['category'] == category
                    assert actual['thresholdRejected'] == bool(rejected[0])
                    assert np.max(np.abs(np.asarray(actual['probabilities']) - probabilities[0])) < 1e-5
                    assert abs(actual['confidence'] - confidence) < 1e-5
                    assert row['displayed_confidence'] == f'{actual["confidence"] * 100:.2f}%'
                    assert row['browser_python_logit_difference'] < 1e-4
                    assert np.max(np.abs(actual_logits - pytorch_logits[0])) < 1e-4
                    if rejected[0]:
                        assert actual['confidenceKind'] == 'highest-target-probability'
                        assert page.locator('#confidence-label').inner_text() == '最高目标类概率'
                        assert '无法确定' in page.locator('#prediction-name').inner_text()
                    else:
                        assert actual['confidenceKind'] == 'predicted-class-probability'
                    if kind == 'lossless-test-pixels': assert tensor_difference <= 1e-7
                    row['status'] = 'PASS'; records.append(row)
                    print(kind, source.name, category, row['displayed_confidence'], 'PASS', flush=True)
            # Actual PNG (RGBA), WebP, and EXIF-orientation paths.
            from PIL import Image
            rgba = load_rgb(chosen[0]).convert('RGBA'); rgba.putalpha(170)
            rgba_path = Path(temp) / 'alpha.png'; rgba.save(rgba_path)
            webp_path = Path(temp) / 'real.webp'; load_rgb(chosen[2]).save(webp_path, lossless=True)
            exif_path = Path(temp) / 'oriented.jpg'
            exif = Image.Exif(); exif[274] = 6
            load_rgb(chosen[0]).save(exif_path, exif=exif)
            format_results = []
            for file in [rgba_path, webp_path, exif_path]:
                page.evaluate('window.lastInference = null')
                page.locator('#file-input').set_input_files(str(file))
                page.wait_for_function("!document.querySelector('#predict-button').disabled")
                page.locator('#predict-button').click()
                page.wait_for_function('window.lastInference !== null')
                format_results.append({'file': file.name, 'status': 'PASS', 'output': page.evaluate('window.lastInference')})
            page.locator('#file-input').set_input_files({'name': 'broken.png', 'mimeType': 'image/png', 'buffer': b'invalid image bytes' * 3})
            page.wait_for_function("!document.querySelector('#error').hidden")
            assert page.locator('#predict-button').is_disabled()
            assert page.locator('#result').is_hidden()
            invalid_message = page.locator('#error').inner_text()
            # Reject a 10 MB+ file before decode.
            page.locator('#file-input').set_input_files({'name': 'oversized.png', 'mimeType': 'image/png', 'buffer': bytes(10 * 1024 * 1024 + 1)})
            page.wait_for_function("document.querySelector('#error').textContent.includes('10 MB')")
            assert page.locator('#predict-button').is_disabled()
            # Confirm an inaccessible real model produces an error, never a prediction.
            failure_page = browser.new_page()
            failure_page.route('**/model.onnx', lambda route: route.abort())
            failure_page.goto(url)
            failure_page.wait_for_function("document.querySelector('#model-status').dataset.state === 'error'", timeout=30000)
            assert failure_page.locator('#predict-button').is_disabled()
            assert failure_page.locator('#result').is_hidden()
            failure_message = failure_page.locator('#error').inner_text()
            failure_page.close()
            page.set_viewport_size({'width': 390, 'height': 844})
            page.reload()
            page.wait_for_function("document.querySelector('#model-status').dataset.state === 'ready'")
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(ROOT / 'results/web_mobile_3class.png'), full_page=True)
            page.set_viewport_size({'width': 1120, 'height': 1050})
            page.screenshot(path=str(ROOT / 'results/web_desktop_3class.png'), full_page=True)
            browser.close()
        assert any(r['browser']['thresholdRejected'] for r in records), 'Actual-image low-confidence branch was not exercised'
        assert any(r['browser']['rawCategory']=='other' for r in records), 'No genuine Other argmax was exercised'
        assert not page_errors, page_errors
        assert all(r['method'] == 'GET' and r['url'].startswith(('http://127.0.0.1:', 'blob:http://127.0.0.1:')) for r in requests), requests
        report = {'status': 'PASS', 'engine': 'real Chromium + ONNX Runtime Web WASM',
                  'model_sha256': sha256(ROOT / 'web/model.onnx'), 'samples': records,
                  'reject_threshold': config['reject_threshold'], 'class_to_idx': saved['class_to_idx'],
                  'real_image_low_confidence_branch_verified': True, 'real_other_argmax_verified': True,
                  'extra_formats': format_results, 'page_errors': page_errors,
                  'invalid_file_error': invalid_message, 'oversized_file_rejected': True,
                  'model_load_error': failure_message, 'all_requests_local_get': True,
                  'mobile_no_horizontal_overflow': True}
        save_json(ROOT / 'results/browser_verification_3class.json', report)
        print('BROWSER VERIFICATION PASS')
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__':
    main()
