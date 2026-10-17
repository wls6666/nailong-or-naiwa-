"""Compare real PyTorch/ONNX logits, embeddings, scores and decisions for all test domains."""
import json
from pathlib import Path
import torch,numpy as np,onnxruntime as ort
from common import ROOT,load_model,save_json,sha256
from dataset import load_rgb,resize_rgb
from ood.core import load_features,distance,energy,probabilities,combined_score


def configured_scores(logits,embedding,config):
 d=distance(embedding,config);e=energy(logits,config['energy_logit_indices'],config['energy_temperature'])
 prob=probabilities(logits)
 method=config['method'];dt=config['mahalanobis_threshold'];et=config['energy_threshold']
 if method=='mahalanobis':score=(d-dt)/config['distance_scale']
 elif method=='energy':score=(e-et)/config['energy_scale']
 else:score=combined_score(d,e,dt,et,config['reject_logic'],config['distance_scale'],config['energy_scale'],prob[:,config['other_logit_index']] if config['other_logit_index'] is not None else None,config['p_other_threshold'])
 return d,e,score


def main():
 torch.set_num_threads(2);selection=json.loads((ROOT/'results/v3/production_selection.json').read_text())['selected'];model,_=load_model(ROOT/selection['checkpoint'])
 _,meta=load_features(ROOT/selection['features'],'test');config=json.loads((ROOT/'web/ood_config.json').read_text());options=ort.SessionOptions();options.intra_op_num_threads=2
 session=ort.InferenceSession(str(ROOT/'web/model.onnx'),sess_options=options,providers=['CPUExecutionProvider']);rows=[]
 for row in meta['rows']:
  tensor=resize_rgb(load_rgb(ROOT/row['path'])).unsqueeze(0)
  with torch.inference_mode():pl,pe=model.forward_outputs(tensor);pl=pl.numpy();pe=pe.numpy()
  ol,oe=session.run(['logits','embedding'],{'input':tensor.numpy()})
  pd,pen,ps=configured_scores(pl,pe,config);od,oen,os=configured_scores(ol,oe,config)
  ld=float(np.max(np.abs(pl-ol)));ed=float(np.max(np.abs(pe-oe)));sd=float(np.max(np.abs(ps-os)))
  passed=bool(ld<1e-4 and ed<1e-4 and np.isfinite(ol).all() and np.isfinite(oe).all() and (ps[0]<=0)==(os[0]<=0))
  rows.append({'path':row['path'],'domain':row['domain'],'logit_max_abs_diff':ld,'embedding_max_abs_diff':ed,'mahalanobis_abs_diff':float(abs(pd[0]-od[0])),'energy_abs_diff':float(abs(pen[0]-oen[0])),'ood_score_abs_diff':sd,'pass':passed})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','count':len(rows),'model_sha256':sha256(ROOT/'web/model.onnx'),'max_logit_difference':max(r['logit_max_abs_diff'] for r in rows),'max_embedding_difference':max(r['embedding_max_abs_diff'] for r in rows),'max_mahalanobis_difference':max(r['mahalanobis_abs_diff'] for r in rows),'max_ood_score_difference':max(r['ood_score_abs_diff'] for r in rows),'samples':rows}
 save_json(ROOT/'results/v3/onnx_verification.json',report);print({k:v for k,v in report.items() if k!='samples'})
 if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
