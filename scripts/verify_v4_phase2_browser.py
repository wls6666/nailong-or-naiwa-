"""Regression-check unchanged production, not a claim of a new V4 deployment."""

import argparse
import sys, json, tempfile, threading
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import onnxruntime as ort
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from common import ROOT, save_json
from dataset import load_rgb
from v4.common import check_production
from verify_ood_onnx import configured_scores


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chromium",
        default="/usr/bin/chromium" if Path("/usr/bin/chromium").exists() else "",
    )
    args = parser.parse_args()
    check_production()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "web"))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = json.loads((ROOT / "web/ood_config.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(ROOT / "web/model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    rows = json.loads(
        (ROOT / "results/v4_phase1/v3_baseline/features/test.json").read_text()
    )["rows"]
    chosen = []
    for label in ["nailong", "naiwa"]:
        chosen.append(next(r for r in rows if r["class"] == label))
    chosen.extend([r for r in rows if r["domain"] == "hard"][:3])
    chosen.extend([r for r in rows if r["domain"] == "easy"][:1])
    results = []
    requests = []
    try:
        with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
            options = {
                "headless": True,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.chromium:
                options["executable_path"] = args.chromium
            browser = pw.chromium.launch(**options)
            page = browser.new_page()
            page.on(
                "request", lambda r: requests.append({"url": r.url, "method": r.method})
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function(
                "document.querySelector('#model-status').dataset.state==='ready'",
                timeout=60000,
            )
            page.evaluate(
                "document.addEventListener('inference-complete',e=>window.actual=e.detail)"
            )
            for i, row in enumerate(chosen):
                png = Path(tmp) / f"probe{i}.png"
                load_rgb(ROOT / row["path"]).save(png)
                page.evaluate("window.actual=null")
                page.locator("#file-input").set_input_files(str(png))
                page.wait_for_function(
                    "!document.querySelector('#predict-button').disabled"
                )
                page.locator("#predict-button").click()
                page.wait_for_function("window.actual!==null")
                a = page.evaluate("window.actual")
                # Compare WASM with native ORT on exactly the browser's own decoded tensor.
                values = page.evaluate(
                    """async()=>{const{imageToCHW}=await import('./preprocess.js');const b=await(await fetch(document.querySelector('#preview').src)).blob();const im=await createImageBitmap(b,{imageOrientation:'from-image',premultiplyAlpha:'none',colorSpaceConversion:'none'});const t=Array.from(imageToCHW(im));im.close();return t;}"""
                )
                tensor = np.array(values, dtype=np.float32).reshape(1, 3, 128, 128)
                l, e = session.run(["logits", "embedding"], {"input": tensor})
                d, en, s = configured_scores(l, e, cfg)
                ld = float(np.max(np.abs(l[0] - a["logits"])))
                ed = float(np.max(np.abs(e[0] - a["embedding"])))
                sd = abs(float(s[0]) - a["oodScore"])
                assert ld < 1e-4 and ed < 1e-4 and sd < 1e-4
                assert a["accepted"] == bool(s[0] <= 0)
                if not a["accepted"]:
                    assert (
                        a["confidence"] is None
                        and "%" not in page.locator("#confidence-value").inner_text()
                    )
                results.append(
                    {
                        "path": row["path"],
                        "category": a["category"],
                        "accepted": a["accepted"],
                        "logits_max_difference": ld,
                        "embedding_max_difference": ed,
                        "ood_score_difference": sd,
                        "status": "PASS",
                    }
                )
            browser.close()
        assert all(
            r["method"] == "GET"
            and r["url"].startswith(("http://127.0.0.1:", "blob:http://127.0.0.1:"))
            for r in requests
        )
        check_production()
        save_json(
            ROOT / "results/v4_phase2/browser_regression.json",
            {
                "scope": "Unchanged production V3 only; no V4 candidate exported/deployed",
                "status": "PASS",
                "real_onnx_wasm_inference": True,
                "samples": results,
                "only_static_local_get_requests": True,
                "production_unchanged": True,
            },
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
