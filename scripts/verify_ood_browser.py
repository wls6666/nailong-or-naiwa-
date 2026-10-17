"""Real static Chromium -> WASM ONNX -> embedding/energy/OOD -> UI verification.
No inference endpoint. PNG pixels isolate decoding, originals test browser codecs.
"""
import argparse,json,tempfile,threading,sys
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch,onnxruntime as ort
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,load_model,save_json,sha256
from dataset import load_rgb,resize_rgb
from ood.core import load_features,probabilities
from verify_ood_onnx import configured_scores

class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--chromium',default='/usr/bin/chromium');args=p.parse_args()
 torch.set_num_threads(2)
 selected=json.loads((ROOT/'results/v3/production_selection.json').read_text())['selected']
 model,_=load_model(ROOT/selected['checkpoint']);config=json.loads((ROOT/'web/ood_config.json').read_text())
 _,meta=load_features(ROOT/selected['features'],'test')
 web_config=json.loads((ROOT/'web/model_config.json').read_text())
 chosen=[]
 for name in ['nailong','naiwa']:chosen.extend([r for r in meta['rows'] if r['class']==name][:2])
 for source in ['cifar100','oxford_pets']:chosen.append(next(r for r in meta['rows'] if r['domain']=='easy' and Path(r['path']).name.startswith(source)))
 for source in ['noto','twemoji','fluent3d']:
  for code in ['1f438','1f432']:chosen.append(next(r for r in meta['rows'] if r.get('source')==source and r.get('source_id')==code))
 options=ort.SessionOptions();options.intra_op_num_threads=2
 session=ort.InferenceSession(str(ROOT/'web/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT/'web')))
 threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'
 requests=[];errors=[];rows=[]
 try:
  with tempfile.TemporaryDirectory() as temp,sync_playwright() as pw:
   opts={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
   if args.chromium:opts['executable_path']=args.chromium
   browser=pw.chromium.launch(**opts);page=browser.new_page(viewport={'width':1120,'height':1050})
   page.on('request',lambda r:requests.append({'url':r.url,'method':r.method}));page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(url);page.wait_for_function("document.querySelector('#model-status').dataset.state === 'ready'",timeout=60000)
   page.evaluate("document.addEventListener('inference-complete',e=>window.lastInference=e.detail)")
   for source_row in chosen:
    source=ROOT/source_row['path'];png=Path(temp)/(source.stem+'.png');load_rgb(source).save(png)
    for image,kind in [(png,'lossless-test-pixels'),(source,'original')]:
     page.evaluate('window.lastInference=null');page.locator('#file-input').set_input_files(str(image));page.wait_for_function("!document.querySelector('#predict-button').disabled")
     page.locator('#predict-button').click();page.wait_for_function('window.lastInference !== null');actual=page.evaluate('window.lastInference')
     tensor_list=page.evaluate('''async()=>{const {imageToCHW}=await import('./preprocess.js'); const blob=await(await fetch(document.querySelector('#preview').src)).blob();const bitmap=await createImageBitmap(blob,{imageOrientation:'from-image',premultiplyAlpha:'none',colorSpaceConversion:'none'});const t=Array.from(imageToCHW(bitmap));bitmap.close();return t;}''')
     tensor=np.asarray(tensor_list,dtype=np.float32).reshape(1,3,128,128);reference=resize_rgb(load_rgb(image)).unsqueeze(0).numpy()
     ol,oe=session.run(['logits','embedding'],{'input':tensor})
     with torch.inference_mode():pl,pe=model.forward_outputs(torch.from_numpy(tensor))
     d,e,score=configured_scores(ol,oe,config);accepted=bool(score[0]<=0);indices=config['energy_logit_indices'];prob=probabilities(ol[:,indices])[0];pred=indices[int(prob.argmax())]
     name=next(k for k,v in config['class_mapping'].items() if v==pred) if accepted else 'other'
     ld=float(np.abs(np.asarray(actual['logits'])-ol[0]).max());ed=float(np.abs(np.asarray(actual['embedding'])-oe[0]).max());td=float(np.abs(tensor-reference).max())
     assert ld<1e-4 and ed<1e-4
     assert float(np.abs(pl.numpy()-ol).max())<1e-4 and float(np.abs(pe.numpy()-oe).max())<1e-4
     assert actual['category']==name and actual['accepted']==accepted
     assert abs(actual['distance']-d[0])<.01 and abs(actual['energy']-e[0])<1e-4 and abs(actual['oodScore']-score[0])<1e-4
     displayed=page.locator('#confidence-value').inner_text()
     if accepted:
      assert abs(actual['confidence']-prob.max())<1e-5 and displayed==f'{actual["confidence"]*100:.2f}%'
      assert page.locator('#confidence-bar').is_visible()
     else:
      assert actual['confidence'] is None and '%' not in displayed and page.locator('#confidence-bar').is_hidden()
     if kind=='lossless-test-pixels':assert td<=1e-7,td
     rows.append({'path':source_row['path'],'domain':source_row['domain'],'kind':kind,'browser':actual,'python_onnx_logits':ol[0].tolist(),'max_preprocessing_tensor_difference':td,'logit_difference':ld,'embedding_difference':ed,'mahalanobis_difference':float(abs(actual['distance']-d[0])),'energy_difference':float(abs(actual['energy']-e[0])),'displayed_confidence':displayed,'status':'PASS'})
     print(kind,source.name,name,'PASS',flush=True)
   # Common real decoding paths: transparent PNG, WebP and EXIF-rotated JPEG.
   from PIL import Image
   rgba=load_rgb(ROOT/chosen[0]['path']).convert('RGBA');rgba.putalpha(170);rgba.save(Path(temp)/'alpha.png')
   load_rgb(ROOT/chosen[2]['path']).save(Path(temp)/'actual.webp',lossless=True)
   exif=Image.Exif();exif[274]=6;load_rgb(ROOT/chosen[0]['path']).save(Path(temp)/'oriented.jpg',exif=exif)
   formats=[]
   for image in [Path(temp)/n for n in ['alpha.png','actual.webp','oriented.jpg']]:
    page.evaluate('window.lastInference=null');page.locator('#file-input').set_input_files(str(image));page.wait_for_function("!document.querySelector('#predict-button').disabled");page.locator('#predict-button').click();page.wait_for_function('window.lastInference !== null');formats.append({'file':image.name,'status':'PASS','category':page.evaluate('window.lastInference.category')})
   for name,data,message in [('broken.png',b'not a valid image'*3,'图片读取失败'),('oversized.png',bytes(10*1024*1024+1),'10 MB')]:
    page.locator('#file-input').set_input_files({'name':name,'mimeType':'image/png','buffer':data});page.wait_for_function('text=>document.querySelector("#error").textContent.includes(text)',arg=message)
    assert page.locator('#result').is_hidden() and page.locator('#predict-button').is_disabled()
   failure_checks=[]
   for resource in [web_config['model'],web_config['ood_config']]:
    failure=browser.new_page();failure.route('**/'+resource,lambda r:r.abort());failure.goto(url);failure.wait_for_function("document.querySelector('#model-status').dataset.state === 'error'",timeout=60000)
    assert failure.locator('#result').is_hidden() and failure.locator('#predict-button').is_disabled();failure_checks.append({'blocked_resource':resource,'message':failure.locator('#error').inner_text()});failure.close()
   # Wrong model bytes must still fail closed; never remove integrity checking.
   corrupted=bytearray((ROOT/'web'/web_config['model']).read_bytes());corrupted[len(corrupted)//2]^=1
   failure=browser.new_page();failure.route('**/'+web_config['model'],lambda r:r.fulfill(status=200,content_type='application/octet-stream',body=bytes(corrupted)));failure.goto(url)
   failure.wait_for_function("document.querySelector('#model-status').dataset.state === 'error'");assert failure.locator('#predict-button').is_disabled() and failure.locator('#result').is_hidden()
   assert '校验不匹配' in failure.locator('#error').inner_text();failure_checks.append({'corrupted_resource':web_config['model'],'message':failure.locator('#error').inner_text()});failure.close()
   # Corrupted parameters must also fail, never fall back to a guessed classification.
   failure=browser.new_page();failure.route('**/'+web_config['ood_config'],lambda r:r.fulfill(status=200,content_type='application/json',body='{}'));failure.goto(url);failure.wait_for_function("document.querySelector('#model-status').dataset.state === 'error'");assert failure.locator('#predict-button').is_disabled();failure_checks.append({'corrupted_resource':'ood_config.json','message':failure.locator('#error').inner_text()});failure.close()
   page.set_viewport_size({'width':390,'height':844});page.reload();page.wait_for_function("document.querySelector('#model-status').dataset.state === 'ready'")
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(ROOT/'results/v3/web_mobile.png'),full_page=True)
   page.set_viewport_size({'width':1120,'height':1050});page.screenshot(path=str(ROOT/'results/v3/web_desktop.png'),full_page=True);browser.close()
  assert any(r['browser']['accepted'] for r in rows) and any(not r['browser']['accepted'] for r in rows)
  assert not errors,errors
  requested_urls=[r['url'] for r in requests]
  assert any('/'+web_config['model'] in u for u in requested_urls)
  assert not any(u.endswith('/model.onnx') or u.endswith('/ood_config.json') for u in requested_urls)
  assert all(r['method']=='GET' and r['url'].startswith(('http://127.0.0.1:','blob:http://127.0.0.1:')) for r in requests)
  save_json(ROOT/'results/v3/browser_verification.json',{'status':'PASS','engine':'Chromium + ONNX Runtime Web WASM (CPU)','model_sha256':sha256(ROOT/'web/model.onnx'),'ood_config_sha256':sha256(ROOT/'web/ood_config.json'),'samples':rows,'formats':formats,'failure_checks':failure_checks,'all_requests_local_get':True,'page_errors':errors,'mobile_no_horizontal_overflow':True,'median_inference_ms':float(np.median([r['browser']['inferenceMs'] for r in rows])),'median_ood_ms':float(np.median([r['browser']['oodMs'] for r in rows]))})
  print('BROWSER OOD VERIFICATION PASS')
 finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
