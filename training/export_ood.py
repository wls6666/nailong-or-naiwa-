"""Export the validation-selected real CNN with logits AND embedding and bound OOD config."""
import json
from pathlib import Path
import torch,onnx,onnxruntime as ort
import numpy as np
from torch import nn
from common import ROOT,load_model,save_json,sha256
from ood.core import load_features,distance,energy


class Outputs(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):return self.model.forward_outputs(x)


def main():
 torch.set_num_threads(2)
 selection=json.loads((ROOT/'results/v3/production_selection.json').read_text());s=selection['selected'];cal=json.loads((ROOT/s['directory']/'calibration.json').read_text());chosen=cal['selected'];p=chosen['parameters'];method=chosen['method']
 if method not in ['mahalanobis','energy','distance_energy','distance_energy_p_other']:raise RuntimeError('Production must genuinely use a validated feature/energy OOD signal')
 model,saved=load_model(ROOT/s['checkpoint']);wrapper=Outputs(model).eval();x=torch.zeros(1,3,128,128)
 with torch.inference_mode():logits,embedding=wrapper(x)
 path=ROOT/'web/model.onnx'
 torch.onnx.export(wrapper,x,str(path),input_names=['input'],output_names=['logits','embedding'],opset_version=17,dynamic_axes={'input':{0:'batch_size'},'logits':{0:'batch_size'},'embedding':{0:'batch_size'}},dynamo=False)
 graph=onnx.load(str(path));onnx.checker.check_model(graph,full_check=True)
 options=ort.SessionOptions();options.intra_op_num_threads=2
 session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
 out=session.run(None,{'input':x.numpy()})
 assert [v.name for v in session.get_outputs()]==['logits','embedding']
 assert out[0].shape==tuple(logits.shape) and out[1].shape==tuple(embedding.shape)
 assert all(np.isfinite(a).all() for a in out)
 gp=json.loads((ROOT/s['directory']/'gaussian_params.json').read_text())
 val,vm=load_features(ROOT/s['features'],'val');is_id=np.asarray([r['domain']=='id' for r in vm['rows']]);indices=[saved['class_to_idx'][n] for n in ['nailong','naiwa']]
 d=distance(val['embedding'],gp)
 # Inactive score settings remain genuine validation ablations for transparent debug values.
 energy_candidates=[r for r in cal['operating_points'] if r['method']=='energy' and r['target_id_accept']==.95]
 ec=min(energy_candidates,key=lambda r:(max(r['easy_ood_far'],r['hard_ood_far']),r['ood_false_accept_rate']))
 temp=p.get('energy_temperature',ec['parameters']['energy_temperature']);e=energy(val['logits'],indices,temp)
 dt=p.get('distance_threshold',float(np.quantile(d[is_id],.95)))
 et=p.get('energy_threshold',ec['parameters']['energy_threshold'])
 ood={'version':'v3','model_sha256':sha256(path),'checkpoint_sha256':sha256(ROOT/s['checkpoint']),
      'method':method,'encoder_family':s['family'],**gp,'class_mapping':{n:saved['class_to_idx'][n] for n in ['nailong','naiwa']},
      'mahalanobis_threshold':dt,'energy_temperature':temp,'energy_threshold':et,'energy_logit_indices':indices,
      'distance_scale':p.get('distance_scale',max(float(np.std(d[is_id])),1e-6)),
      'energy_scale':p.get('energy_scale',max(float(np.std(e[is_id])),1e-6)),
      'reject_logic':p.get('reject_logic','or'),'p_other_threshold':p.get('p_other_threshold'),
      'other_logit_index':saved['class_to_idx'].get('other'),
      'active_signals':(['distance'] if method=='mahalanobis' else ['energy'] if method=='energy' else ['distance','energy'])+(['p_other'] if method.endswith('p_other') else []),
      'calibration_split':'val','calibration_sha256':sha256(ROOT/s['directory']/'calibration.json'),
      'score_semantics':'signed normalized distance/energy margin; >0 is reject; NOT a probability'}
 save_json(ROOT/'web/ood_config.json',ood)
 config={'task':'open-set-ood-v3','model':'model.onnx','input_name':'input','output_names':['logits','embedding'],
         'input_shape':[1,3,128,128],'logits_shape':list(logits.shape),'embedding_shape':list(embedding.shape),
         'class_to_idx':saved['class_to_idx'],'classes':[n for n,i in sorted(saved['class_to_idx'].items(),key=lambda kv:kv[1])],
         'preprocessing':saved['preprocessing'],'model_sha256':sha256(path),'checkpoint_sha256':sha256(ROOT/s['checkpoint']),
         'ood_config':'ood_config.json','ood_config_sha256':sha256(ROOT/'web/ood_config.json')}
 save_json(ROOT/'web/model_config.json',config)
 save_json(ROOT/'results/v3/onnx_export.json',{'onnx_checker':'PASS','python_runtime':'PASS','model_bytes':path.stat().st_size,'ood_config_bytes':(ROOT/'web/ood_config.json').stat().st_size,**config})
 from publish_web_assets import publish
 publish()
 print('Export PASS',s['family'],'shapes',list(logits.shape),list(embedding.shape),'ONNX bytes',path.stat().st_size,'OOD JSON bytes',(ROOT/'web/ood_config.json').stat().st_size)


if __name__=='__main__':main()
