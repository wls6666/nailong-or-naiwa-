"""Export exactly the validation-selected MobileNet candidate, never production.
Verify actual test logits/embeddings across ID/Easy/Hard against Python ORT.
"""
import json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch,onnx,onnxruntime as ort
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save_json,sha256
from v4.common import BASE,load,Images,rows,check_production
from v4.detector import scores
from torch.utils.data import DataLoader

class Outputs(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):return self.model.forward_outputs(x)


def main():
 torch.set_num_threads(2);check_production();selection=json.loads((BASE/'candidate_selection.json').read_text());name=selection['candidate']['name'];model,size,path=load(name)
 assert sha256(path)==selection['candidate']['checkpoint_sha256']
 out=ROOT/'artifacts/v4';out.mkdir(parents=True,exist_ok=True);model_path=out/'model.onnx';dummy=torch.zeros(1,3,size,size)
 torch.onnx.export(Outputs(model).eval(),dummy,str(model_path),input_names=['input'],output_names=['logits','embedding'],opset_version=17,dynamo=False,dynamic_axes={'input':{0:'batch_size'},'logits':{0:'batch_size'},'embedding':{0:'batch_size'}})
 onnx.checker.check_model(onnx.load(str(model_path)),full_check=True)
 cfg=json.loads((BASE/name/'ood_config.json').read_text());cfg['model_sha256']=sha256(model_path);save_json(out/'ood_config.json',cfg)
 model_config={'task':'v4-phase1-experiment-only','candidate':name,'model':'model.onnx','model_sha256':sha256(model_path),'checkpoint_sha256':sha256(path),'input_name':'input','output_names':['logits','embedding'],'input_shape':[1,3,size,size],'logits_shape':[1,2],'embedding_shape':[1,128],'class_to_idx':{'nailong':0,'naiwa':1},'preprocessing':{'size':size,'channels':'RGB','layout':'NCHW','dtype':'float32','range':[0,1],'resize':'bilinear-half-pixel-clamp-float-no-round','alpha_background':'white','normalization':'ImageNet mean/std inside ONNX graph; external input remains0..1'},'ood_config':'ood_config.json','ood_config_sha256':sha256(out/'ood_config.json'),'production_replaced':False}
 save_json(out/'model_config.json',model_config)
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
 start=time.perf_counter();session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider']);load_seconds=time.perf_counter()-start
 assert [o.name for o in session.get_outputs()]==['logits','embedding']
 ds=Images(rows('test'),size);records=[];offset=0;max_l=max_e=0.;all_decisions=True
 with torch.inference_mode():
  for x,y in DataLoader(ds,batch_size=16):
   pl,pe=model.forward_outputs(x);ol,oe=session.run(None,{'input':x.numpy()});ld=np.abs(pl.numpy()-ol).max(1);ed=np.abs(pe.numpy()-oe).max(1)
   ps=scores(pl.numpy(),pe.numpy(),cfg)[2];os=scores(ol,oe,cfg)[2]
   for j in range(len(y)):
    same=bool((ps[j]<=0)==(os[j]<=0));passed=bool(ld[j]<1e-4 and ed[j]<1e-4 and same);all_decisions &= passed
    records.append({'path':ds.rows[offset+j]['path'],'domain':ds.rows[offset+j]['domain'],'logit_max_abs_diff':float(ld[j]),'embedding_max_abs_diff':float(ed[j]),'same_decision':same,'pass':passed})
   offset+=len(y);max_l=max(max_l,float(ld.max()));max_e=max(max_e,float(ed.max()))
 # Actual single-image CPU timing, warmup + repeat; data-dependent image, not zero demo.
 tensor=ds[0][0].unsqueeze(0).numpy()
 for _ in range(5):session.run(None,{'input':tensor})
 times=[]
 for _ in range(30):
  t=time.perf_counter();session.run(None,{'input':tensor});times.append((time.perf_counter()-t)*1000)
 report={'status':'PASS' if all_decisions else 'FAIL','candidate':name,'count':len(records),'onnx_checker':'PASS','output_names':[o.name for o in session.get_outputs()],'model_sha256':sha256(model_path),'checkpoint_bytes':path.stat().st_size,'onnx_bytes':model_path.stat().st_size,'browser_download_model_bytes':model_path.stat().st_size,'input_size':size,'max_logit_abs_difference':max_l,'max_embedding_abs_difference':max_e,'cpu_onnx_load_seconds':load_seconds,'cpu_onnx_median_ms':float(np.median(times)),'cpu_onnx_timing_threads':2,'warmup':5,'timing_runs':30,'samples':records,'production_unchanged':True}
 save_json(BASE/'onnx_verification.json',report);check_production();print({k:v for k,v in report.items() if k!='samples'},flush=True)
 if not all_decisions:raise RuntimeError('ONNX parity failed; see actual differences')

if __name__=='__main__':main()
