"""Real browser regression for immutable model paths and fail-closed retry.
Fault responses are deliberate network tests, never substitute model predictions.
"""
import json,sys,threading
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,save_json,sha256

class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def main():
 config=json.loads((ROOT/'web/model_config.json').read_text());bytes_model=(ROOT/'web'/config['model']).read_bytes()
 meta=json.loads((ROOT/'results/v3/phase_b_features/test.json').read_text());image=ROOT/next(r['path'] for r in meta['rows'] if r['class']=='nailong')
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT/'web')));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'
 records=[]
 def wait_state(page,state):
  page.wait_for_function('state=>document.querySelector("#model-status").dataset.state===state',arg=state,timeout=60000)
 def predict(page):
  page.evaluate("window.actualInference=null;document.addEventListener('inference-complete',e=>window.actualInference=e.detail,{once:true})")
  page.locator('#file-input').set_input_files(str(image));page.wait_for_function("!document.querySelector('#predict-button').disabled");page.locator('#predict-button').click();page.wait_for_function('window.actualInference!==null')
  actual=page.evaluate('window.actualInference');assert actual['outputShape']==[1,2] and actual['embeddingShape']==[1,256]
  return {k:actual[k] for k in ['category','accepted','logits','distance','energy','oodScore']}
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
   page=browser.new_page();requests=[];page.on('request',lambda r:requests.append(r.url))
   # A stale legacy path must not affect the fingerprinted real CNN request.
   page.route('**/model.onnx',lambda r:r.fulfill(status=200,content_type='application/octet-stream',body=(ROOT/'checkpoints/v2/model.onnx').read_bytes()))
   page.goto(url);wait_state(page,'ready');actual=predict(page)
   assert any(u.endswith('/'+config['model']) for u in requests) and not any(u.endswith('/model.onnx') for u in requests)
   records.append({'case':'stale legacy model URL bypassed','status':'PASS','real_inference':actual,'model_url':config['model']});page.close()
   corrupted=bytearray(bytes_model);corrupted[len(corrupted)//2]^=1
   for case,payload,expected in [('same-size corrupt model',bytes(corrupted),'校验不匹配'),('incomplete model',bytes_model[:512],'模型下载不完整')]:
    page=browser.new_page();pattern='**/'+config['model'];page.route(pattern,lambda r:r.fulfill(status=200,content_type='application/octet-stream',body=payload));page.goto(url);wait_state(page,'error')
    message=page.locator('#error').inner_text();assert expected in message and page.locator('#predict-button').is_disabled() and page.locator('#result').is_hidden()
    # A successful retry must download and execute the genuine model.
    page.unroute(pattern);page.locator('#retry-button').click();wait_state(page,'ready');actual=predict(page)
    records.append({'case':case,'status':'PASS','error':message,'retry_real_inference':actual});page.close()
   browser.close()
  save_json(ROOT/'results/v3/asset_delivery_verification.json',{'status':'PASS','model_sha256':sha256(ROOT/'web'/config['model']),'ood_config_sha256':sha256(ROOT/'web'/config['ood_config']),'model_bytes':len(bytes_model),'cases':records,'production_root_cause':'Unconfirmed: production domain returned HTTP 403/1010 from this environment. Local/GitHub canonical bytes match the manifest; regression covers stale canonical path, wrong bytes, truncation and genuine-inference retry.'})
  print('PASS: fingerprinted real model, integrity failure, incomplete response and real-inference retry')
 finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
