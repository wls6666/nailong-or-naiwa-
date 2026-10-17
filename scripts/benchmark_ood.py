"""Compare real custom/MobileNet ONNX size and local WASM speed after selection.
Timing only, no tuning or new accuracy measurement; small CPU sample, not device guarantee.
"""
import json,sys,threading
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import torch,numpy as np
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,load_model,save_json
from dataset import resize_rgb,load_rgb
from export_ood import Outputs

def main():
 torch.set_num_threads(2);out=ROOT/'data/benchmark';out.mkdir(parents=True,exist_ok=True)
 model,_=load_model(ROOT/'checkpoints/mobilenet_ood.pth')
 torch.onnx.export(Outputs(model).eval(),torch.zeros(1,3,128,128),str(out/'mobilenet.onnx'),input_names=['input'],output_names=['logits','embedding'],opset_version=17,dynamo=False)
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*a):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT)));threading.Thread(target=server.serve_forever,daemon=True).start()
 source=json.loads((ROOT/'results/v3/phase_b_features/test.json').read_text())['rows'][0]['path'];tensor=resize_rgb(load_rgb(ROOT/source)).numpy().reshape(-1).tolist()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page();page.goto(f'http://127.0.0.1:{server.server_port}/web/');page.wait_for_function("document.querySelector('#model-status').dataset.state==='ready'")
   records=page.evaluate('''async data=>{const records=[];for(const [name,url] of [['Custom CNN + OE','/web/model.onnx'],['MobileNetV3-Small pretrained frozen + PCA128','/data/benchmark/mobilenet.onnx']]){const bytes=await(await fetch(url)).arrayBuffer();const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});const tensor=new ort.Tensor('float32',Float32Array.from(data),[1,3,128,128]);const times=[];for(let n=0;n<13;n++){const start=performance.now();const outputs=await session.run({input:tensor});const ms=performance.now()-start;if(n>=3)times.push(ms);if(!Array.from(outputs.logits.data).every(Number.isFinite))throw Error('Nonfinite inference');}times.sort((a,b)=>a-b);records.push({name,model_bytes:bytes.byteLength,median_ms:(times[4]+times[5])/2,warmup:3,measured_runs:10});await session.release();}return records;}''',tensor);browser.close()
  save_json(ROOT/'results/v3/runtime_comparison.json',{'engine':'Chromium WASM, single thread, this CPU environment only','input_image':source,'selection_affected':False,'models':records});print(records)
 finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
