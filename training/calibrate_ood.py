"""Fit only ID train; compare/calibrate only validation; never reads test."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,save_json,sha256
from ood.core import fit_gaussian,distance,energy,probabilities,evaluate_accept,combined_score,load_features


def quantiles(values):
 return np.unique(np.r_[np.quantile(values,np.r_[np.linspace(0,1,31),.90,.92,.94,.95,.96,.97,.98,.99]),np.nextafter(np.max(values),np.inf)])


def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=ROOT/'results/v3/features');p.add_argument('--output',type=Path,default=ROOT/'results/v3/phase_a');args=p.parse_args()
 train,tm=load_features(args.features,'train');val,vm=load_features(args.features,'val')
 if tm['checkpoint_sha256']!=vm['checkpoint_sha256']:raise RuntimeError('Feature checkpoint mismatch')
 mapping=tm['class_to_idx'];indices=[mapping['nailong'],mapping['naiwa']]
 rows=vm['rows'];domains=np.asarray([r['domain'] for r in rows]);is_id=domains=='id'
 masks={k:domains==k for k in ['id','easy','hard']};has_hard=masks['hard'].any()
 prob=probabilities(val['logits']);pother=prob[:,mapping['other']] if val['logits'].shape[1]==3 else None;msp=1-probabilities(val['logits'][:,indices]).max(1)
 out=[];best_params={};diagnostics=[]
 # Threshold calibration ranks feasible methods by worst-domain FAR, then pooled FAR,
 # preserving ID >= each requested operating point, then end-to-end ID accuracy.
 def key(rec):
  fars=[rec[f'{x}_ood_far'] for x in ['easy','hard'] if rec[f'{x}_ood_far'] is not None]
  return (max(fars),rec['ood_false_accept_rate'],-rec['id_end_to_end_accuracy'],-rec['id_true_accept_rate'])
 def add(name,score,accepted,params,target):
  rec={'method':name,'target_id_accept':target,**evaluate_accept(accepted,rows,val['logits'],indices,score),'parameters':params}
  return rec
 # Immutable old deployed policy, and individually calibrated scalar baselines.
 oldscore=1-prob[:,indices].max(1)
 if pother is not None:out.append(add('v2_softmax',oldscore,oldscore<=.45,{'threshold':.45,'score':'1-max_target_softmax_3'},None))
 d_variants={}
 for normalized in [False,True]:
  gp=fit_gaussian(train['embedding'],train['labels'],indices,normalized)
  d=distance(val['embedding'],gp);d_variants[str(normalized)]=(gp,d)
  diag=evaluate_accept(np.ones(len(rows),dtype=bool),rows,val['logits'],indices,d)
  diagnostics.append({'normalize_embedding':normalized,'shrinkage':gp['shrinkage'],'auroc':diag['auroc'],'easy_auroc':diag['easy_ranking']['auroc'],'hard_auroc':diag['hard_ranking']['auroc']})
 for target in [.92,.95,.97,.98]:
  scalar_inputs=[('msp',msp,{'score':'1-max_target_softmax_2'})]
  if pother is not None:scalar_inputs.append(('p_other',pother,{'score':'p_other'}))
  for name,scores,meta in scalar_inputs:
   candidates=[add(name,scores,scores<=t,{**meta,'threshold':float(t)},target) for t in quantiles(scores[is_id]) if np.mean(scores[is_id]<=t)>=target]
   out.append(min(candidates,key=key))
  for normalized_key,(gp,d) in d_variants.items():
   normalized=gp['normalize_embedding']
   ds=max(float(np.std(d[is_id])),1e-6)
   ds_grid=quantiles(d[is_id])
   candidates=[add('mahalanobis',d,d<=dt,{'normalize_embedding':normalized,'distance_threshold':float(dt)},target) for dt in ds_grid if np.mean(d[is_id]<=dt)>=target]
   best_d=min(candidates,key=key);out.append(best_d)
   for temp in [.5,1.,2.,5.]:
    e=energy(val['logits'],indices,temp);es=max(float(np.std(e[is_id])),1e-6);es_grid=quantiles(e[is_id])
    candidates=[add('energy',e,e<=et,{'energy_temperature':temp,'energy_threshold':float(et)},target) for et in es_grid if np.mean(e[is_id]<=et)>=target]
    # Energy is independent of embedding variant, record it once.
    if not normalized:out.append(min(candidates,key=key))
    for logic in ['or','and']:
     best=None;best_extra=None
     for dt in ds_grid:
      for et in es_grid:
       accepted=((d<=dt)&(e<=et)) if logic=='or' else ((d<=dt)|(e<=et))
       if accepted[is_id].mean()+1e-12<target:continue
       # Fast candidate ranking; full AUROC computed only for winning combinations.
       fars=[accepted[mask].mean() for name,mask in masks.items() if name!='id' and mask.any()]
       pred=np.asarray(indices)[val['logits'][:,indices].argmax(1)]
       rank=(max(fars),accepted[~is_id].mean(),-((pred==val['labels'])&accepted)[is_id].mean(),-accepted[is_id].mean())
       params={'normalize_embedding':normalized,'distance_threshold':float(dt),'energy_threshold':float(et),'energy_temperature':temp,'reject_logic':logic,'distance_scale':ds,'energy_scale':es}
       if best is None or rank<best[0]:best=(rank,params,accepted.copy())
       # A modest p_other grid, calibrated from validation only; never default to adding it.
       for pt in ([.2,.4,.6,.8,.95,1.] if pother is not None else []):
        ac=accepted&(pother<=pt)
        if ac[is_id].mean()+1e-12<target:continue
        fs=[ac[mask].mean() for name,mask in masks.items() if name!='id' and mask.any()]
        rk=(max(fs),ac[~is_id].mean(),-((pred==val['labels'])&ac)[is_id].mean(),-ac[is_id].mean())
        if best_extra is None or rk<best_extra[0]:best_extra=(rk,{**params,'p_other_threshold':pt},ac.copy())
     for name,b in [('distance_energy',best),('distance_energy_p_other',best_extra)]:
      if b:
       _,params,accepted=b;score=combined_score(d,e,params['distance_threshold'],params['energy_threshold'],logic,ds,es,pother,params.get('p_other_threshold'))
       out.append(add(name,score,accepted,params,target))
 # Select the balanced >=95% operating point. Prefer distance+energy unless an
 # ablation is strictly better; tie favour simpler feature-only over redundant signals.
 feasible=[r for r in out if r['target_id_accept']==.95]
 simplicity={'mahalanobis':0,'energy':1,'distance_energy':2,'distance_energy_p_other':3,'p_other':4,'msp':5}
 selected=min(feasible,key=lambda r:(*key(r),simplicity[r['method']]))
 gp=d_variants[str(selected['parameters'].get('normalize_embedding',False))][0]
 save_json(args.output/'gaussian_params.json',gp)
 report={'phase':'post-hoc-calibration','calibration_split':'val','checkpoint_sha256':tm['checkpoint_sha256'],'class_mapping':mapping,
         'embedding_comparison':diagnostics,'operating_points':out,'selected':selected,
         'selection_rule':'Default ID accept >=95%; minimize worst Easy/Hard FAR, then pooled FAR, then preserve correct ID and total ID; exact ties prefer simpler methods. Test is never read.',
         'hard_validation_count':int(masks['hard'].sum()),'gaussian_params_sha256':sha256(args.output/'gaussian_params.json')}
 save_json(args.output/'calibration.json',report)
 print('Raw vs L2:',diagnostics,flush=True)
 print('SELECTED',json.dumps(selected,indent=2),flush=True)
 for name in ['v2_softmax','msp','p_other','mahalanobis','energy','distance_energy','distance_energy_p_other']:
  recs=[r for r in out if r['method']==name and r['target_id_accept'] in [None,.95]]
  if recs:
   r=min(recs,key=key);print(name,'TAR',r['id_true_accept_rate'],'easy FAR',r['easy_ood_far'],'hard FAR',r['hard_ood_far'],'AUROC',r['auroc'],flush=True)


if __name__=='__main__':main()
