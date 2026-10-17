"""Experimental candidate + untouched V3, real browser WASM and static GET only.
No new product UI. Preprocessing, embedding/logits and OOD math are truly run.
"""
import sys,json,time,threading,argparse
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import numpy as np,torch,onnxruntime as ort
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,save_json,sha256
from dataset import load_rgb
from v4.common import BASE,load,resize,rows,check_production
from v4.detector import scores

class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium' if Path('/usr/bin/chromium').exists() else '',help='Empty string uses Playwright-managed Chromium');args=parser.parse_args()
 torch.set_num_threads(2);check_production();selection=json.loads((BASE/'candidate_selection.json').read_text());candidate=selection['candidate']['name'];allrows=rows('test');chosen=[]
 for group in ['nailong','naiwa']:
  chosen.extend([r for r in allrows if r['class']==group][:2])
 chosen.extend([r for r in allrows if r['domain']=='easy'][:2])
 for source in ['noto','twemoji','fluent3d']:chosen.append(next(r for r in allrows if r['domain']=='hard' and r['source']==source))
 parity=json.loads((BASE/'onnx_verification.json').read_text());worst=max(parity['samples'],key=lambda r:r['embedding_max_abs_diff']);chosen.append(next(r for r in allrows if r['path']==worst['path']));chosen=list({r['path']:r for r in chosen}.values())
 probes=ROOT/'data/v4_browser_probe';probes.mkdir(exist_ok=True)
 for i,r in enumerate(chosen):load_rgb(ROOT/r['path']).save(probes/f'{i}.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT)));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/';reports=[];requests=[]
 try:
  with sync_playwright() as pw:
   options={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
   if args.chromium:options['executable_path']=args.chromium
   browser=pw.chromium.launch(**options)
   for name in ['v3_baseline',candidate]:
    model,size,checkpoint=load(name);where='web' if name=='v3_baseline' else 'artifacts/v4';config=json.loads((ROOT/where/'model_config.json').read_text());ood=json.loads((ROOT/where/config['ood_config']).read_text())
    page=browser.new_page();page.on('request',lambda r:requests.append({'url':r.url,'method':r.method}));page.goto(url+'artifacts/v4/');page.add_script_tag(url=url+'web/assets/vendor/onnxruntime-web/ort.wasm.min.js')
    setup=page.evaluate('''async args=>{ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('/web/assets/vendor/onnxruntime-web/',location.href).href;const start=performance.now();const response=await fetch('/'+args.folder+'/'+args.config.model);if(!response.ok)throw Error('model HTTP '+response.status);const bytes=await response.arrayBuffer();const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');if(hash!==args.config.model_sha256)throw Error('model hash mismatch');const downloaded=performance.now();window.session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});return {model_bytes:bytes.byteLength,download_ms:downloaded-start,session_create_ms:performance.now()-downloaded,initial_load_ms:performance.now()-start,inputs:session.inputNames,outputs:session.outputNames};}''',{'folder':where,'config':config})
    assert setup['outputs']==['logits','embedding']
    options=ort.SessionOptions();options.intra_op_num_threads=2;session=ort.InferenceSession(str(ROOT/where/config['model']),sess_options=options,providers=['CPUExecutionProvider']);cases=[]
    for i,r in enumerate(chosen):
     actual=page.evaluate('''async args=>{const {imageToCHW}=await import('/artifacts/v4/preprocess.js');const {inferOOD}=await import('/web/ood.js');const blob=await(await fetch('/data/v4_browser_probe/'+args.index+'.png')).blob();const bitmap=await createImageBitmap(blob,{imageOrientation:'from-image',premultiplyAlpha:'none',colorSpaceConversion:'none'});const data=imageToCHW(bitmap,args.size);bitmap.close();const tensor=new ort.Tensor('float32',data,[1,3,args.size,args.size]);const start=performance.now();const outputs=await session.run({input:tensor});const inferenceMs=performance.now()-start;const logits=Array.from(outputs.logits.data),embedding=Array.from(outputs.embedding.data);const t=performance.now();const decision=inferOOD(logits,embedding,args.ood);const oodMs=performance.now()-t;window.probeTensor=tensor;return {tensor:Array.from(data),logits,embedding,decision,inferenceMs,oodMs};}''',{'index':i,'size':size,'ood':ood})
     tensor=np.asarray(actual.pop('tensor'),dtype=np.float32).reshape(1,3,size,size);reference=resize(load_rgb(probes/f'{i}.png'),size).unsqueeze(0).numpy();td=float(np.abs(tensor-reference).max());assert td<=1e-7
     ol,oe=session.run(None,{'input':tensor})
     with torch.inference_mode():pl,pe=model.forward_outputs(torch.from_numpy(tensor))
     ld=float(np.abs(np.asarray(actual['logits'])-ol[0]).max());ed=float(np.abs(np.asarray(actual['embedding'])-oe[0]).max())
     assert ld<1e-4 and ed<1e-4,(name,ld,ed)
     assert float(np.abs(pl.numpy()-ol).max())<1e-4 and float(np.abs(pe.numpy()-oe).max())<1e-4
     # Match JavaScript quadratic-form/energy arithmetic on EXACT same output values.
     d,e,s=scores(np.asarray([actual['logits']]),np.asarray([actual['embedding']]),ood);decision=actual['decision']
     assert np.allclose([decision['distance'],decision['energy'],decision['oodScore']],[d[0],e[0],s[0]],atol=1e-6,rtol=1e-9)
     expected=scores(ol,oe,ood)[2];assert decision['accepted']==bool(expected[0]<=0)
     if not decision['accepted']:assert decision['confidence'] is None
     cases.append({'path':r['path'],'domain':r['domain'],'status':'PASS','max_tensor_difference':td,'browser_onnx_logit_difference':ld,'browser_onnx_embedding_difference':ed,**actual})
    timing=page.evaluate('''async()=>{for(let n=0;n<5;n++)await session.run({input:probeTensor});const times=[];for(let n=0;n<20;n++){const t=performance.now();await session.run({input:probeTensor});times.push(performance.now()-t);}times.sort((a,b)=>a-b);return {warmup:5,runs:20,median_ms:(times[9]+times[10])/2};}''')
    reports.append({'name':name,'input_size':size,'status':'PASS','initial_load':setup,'browser_timing':timing,'median_ood_ms':float(np.median([r['oodMs'] for r in cases])),'cases':cases});page.close()
   browser.close()
  assert all(r['method']=='GET' and r['url'].startswith(url) for r in requests)
  check_production();save_json(BASE/'browser_verification.json',{'status':'PASS','engine':'Chromium + ORT Web 1.22 WASM single thread, local CPU HTTP','models':reports,'all_requests_static_local_get':True,'production_unchanged':True,'timing_limitation':'Fresh page/session per model; runtime load includes WASM startup. Local network, single CPU machine, not representative of Internet download/phone speed.'});print('V4 BROWSER PASS',[(r['name'],r['browser_timing'],r['initial_load']['initial_load_ms']) for r in reports])
 finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
