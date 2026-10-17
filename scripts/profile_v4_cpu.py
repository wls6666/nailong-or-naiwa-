"""Measure the frozen V3 and one exported candidate, same real ID image, CPU ORT."""
import sys,json,time
from pathlib import Path
import numpy as np,onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,save_json
from dataset import load_rgb
from v4.common import BASE,rows,resize,load,check_production
candidate=json.loads((BASE/'candidate_selection.json').read_text())['candidate']['name'];browser=json.loads((BASE/'browser_verification.json').read_text());source=rows('test')[0]['path'];records=[]
for name in ['v3_baseline',candidate]:
 where=ROOT/('web' if name=='v3_baseline' else 'artifacts/v4');cfg=json.loads((where/'model_config.json').read_text());size=cfg['input_shape'][2];model,size,checkpoint=load(name)
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
 start=time.perf_counter();session=ort.InferenceSession(str(where/cfg['model']),sess_options=options,providers=['CPUExecutionProvider']);load_ms=(time.perf_counter()-start)*1000
 x=resize(load_rgb(ROOT/source),size).unsqueeze(0).numpy()
 for _ in range(5):session.run(None,{'input':x})
 times=[]
 for _ in range(30):
  start=time.perf_counter();session.run(None,{'input':x});times.append((time.perf_counter()-start)*1000)
 b=next(r for r in browser['models'] if r['name']==name)
 records.append({'name':name,'input_size':size,'checkpoint_bytes':checkpoint.stat().st_size,'onnx_bytes':(where/cfg['model']).stat().st_size,'browser_model_bytes':b['initial_load']['model_bytes'],'cpu_onnx_load_ms':load_ms,'cpu_onnx_median_ms':float(np.median(times)),'browser_wasm_median_ms':b['browser_timing']['median_ms'],'browser_initial_load_ms':b['initial_load']['initial_load_ms'],'browser_ood_median_ms':b['median_ood_ms']})
check_production();save_json(BASE/'performance.json',{'models':records,'input_image':source,'cpu_threads':2,'cpu_warmup':5,'cpu_runs':30,'browser_wasm_threads':1,'limitations':'This CPU/local HTTP only; not Internet download or phone speed. Only validation-selected MobileNet exported.'});print(records)
