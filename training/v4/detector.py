"""Validation-only detector selection and frozen evaluation helpers."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from common import ROOT,save_json,sha256
from ood.core import fit_gaussian,distance,energy,combined_score,probabilities,load_features
from v4.common import BASE


def key(r):return (max(r['easy_ood_far'],r['hard_ood_far']),r['ood_false_accept_rate'],-r['id_end_to_end_accuracy'],-r['id_true_accept_rate'])


def calibrate(name):
 folder=BASE/name;out=folder/'ood';out.mkdir(parents=True,exist_ok=True)
 with (folder/'calibration.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'training/calibrate_ood.py'),'--features',str(folder/'features'),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
 cal=json.loads((out/'calibration.json').read_text())
 candidates=[r for r in cal['operating_points'] if r['target_id_accept']==.95 and r['method'] in ['mahalanobis','energy','distance_energy']]
 simplicity={'mahalanobis':0,'energy':1,'distance_energy':2}
 selected=min(candidates,key=lambda r:(*key(r),simplicity[r['method']]))
 cal['selected']=selected;cal['phase']='V4 validation-only OOD calibration';cal['protocol_sha256']=sha256(ROOT/'experiments/v4_phase1/protocol.json')
 train,_=load_features(folder/'features','train');val,vm=load_features(folder/'features','val');p=selected['parameters']
 gp=fit_gaussian(train['embedding'],train['labels'],[0,1],p.get('normalize_embedding',False));save_json(out/'gaussian_params.json',gp);cal['gaussian_params_sha256']=sha256(out/'gaussian_params.json');save_json(out/'calibration.json',cal)
 idmask=np.asarray([r['domain']=='id' for r in vm['rows']]);d=distance(val['embedding'],gp)
 erow=min((r for r in candidates if r['method']=='energy'),key=key);temp=p.get('energy_temperature',erow['parameters']['energy_temperature']);e=energy(val['logits'],[0,1],temp)
 config={**gp,'version':'v4-phase1','method':selected['method'],'class_mapping':{'nailong':0,'naiwa':1},'energy_logit_indices':[0,1],
  'mahalanobis_threshold':p.get('distance_threshold',float(np.quantile(d[idmask],.95))), 'energy_temperature':temp,'energy_threshold':p.get('energy_threshold',erow['parameters']['energy_threshold']),
  'distance_scale':p.get('distance_scale',max(float(np.std(d[idmask])),1e-6)),'energy_scale':p.get('energy_scale',max(float(np.std(e[idmask])),1e-6)),
  'reject_logic':p.get('reject_logic','or'),'other_logit_index':None,'p_other_threshold':None,
  'calibration_split':'val','checkpoint_sha256':cal['checkpoint_sha256'],'calibration_sha256':sha256(out/'calibration.json')}
 save_json(folder/'ood_config.json',config)
 print('CALIBRATED',name,selected['method'],'val FAR',selected['easy_ood_far'],selected['hard_ood_far'],'ID accept',selected['id_true_accept_rate'],flush=True)
 return cal


def scores(logits,embedding,config):
 d=distance(embedding,config);e=energy(logits,config['energy_logit_indices'],config['energy_temperature'])
 if config['method']=='mahalanobis':s=(d-config['mahalanobis_threshold'])/config['distance_scale']
 elif config['method']=='energy':s=(e-config['energy_threshold'])/config['energy_scale']
 else:s=combined_score(d,e,config['mahalanobis_threshold'],config['energy_threshold'],config['reject_logic'],config['distance_scale'],config['energy_scale'])
 return d,e,s
