"""Check provenance, fixed-set metrics, actual training selection, parity and V3 preservation."""
import json,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,sha256
from v4.common import BASE,check_production
from v4.detector import key

def read(path):return json.loads((ROOT/path).read_text())
def main():
 check_production();lock=read('experiments/v4_phase1/data_lock.json');manifest=read('data/hard_ood_test_manifest.json')
 assert lock['protocol_sha256']==sha256(ROOT/'experiments/v4_phase1/protocol.json')
 assert lock['benchmark_sha256']==sha256(ROOT/'data/hard_ood_test_manifest.json')==sha256(BASE/'hard_ood_test_manifest.json')
 assert len(manifest['samples'])==manifest['sample_count']==321 and manifest['identity_count']==121
 old=read('data/hard_ood_manifest.json');assert not {r['identity_group'] for r in old['samples']}&{r['identity_group'] for r in manifest['samples']}
 for r in manifest['samples']:assert not r['used_for_training'] and not r['used_for_validation'] and sha256(ROOT/r['path'])==r['file_sha256']
 selection=read('results/v4_phase1/candidate_selection.json');assert selection['test_used'] is False and selection['selection_split']=='val' and selection['data_lock']==lock
 eligible=[r for r in selection['experiments'] if r['eligible_id']];assert selection['candidate']==min(eligible or selection['experiments'],key=lambda r:key(r['validation']))
 names=['v3_baseline','mobilenet_frozen','mobilenet_partial_ft','mobilenet_frozen_128_control'];sets=[]
 for name in names:
  folder=BASE/name;metrics=json.loads((folder/'metrics.json').read_text());prediction=list(csv.DictReader((folder/'predictions.csv').open()));assert len(prediction)==metrics['count']==544
  md=json.loads((folder/'features/test.json').read_text());assert md['set_sha256']==metrics['test_set_sha256']==lock['test_set_sha256'];sets.append([r['path'] for r in md['rows']])
  assert metrics['test_used_to_tune'] is False
  for domain,n in [('id',73),('easy',150),('hard',321)]:
   rows=[r for r in prediction if r['domain']==domain];assert len(rows)==n
   rate=sum(r['accepted']=='True' for r in rows)/n
   assert abs(rate-metrics['id_true_accept_rate' if domain=='id' else domain+'_ood_far'])<1e-12
  if name=='v3_baseline':continue
  training=json.loads((folder/'training_summary.json').read_text());history=json.loads((folder/'training_history.json').read_text());best=max(history,key=lambda r:(r['val_accuracy'],-r['val_loss']))
  assert best['epoch']==training['best_epoch'] and training['test_read'] is False
  assert sha256(ROOT/training['checkpoint'])==training['checkpoint_sha256']==metrics['checkpoint_sha256']
  config=json.loads((folder/'training_config.json').read_text());assert config['protocol_sha256']==lock['protocol_sha256'] and config['seed']==42 and config['train_count']==345 and config['val_count']==73
  for split in ['train','val']:
   meta=json.loads((folder/f'features/{split}.json').read_text());assert meta['set_sha256']==lock[split+'_set_sha256']
   assert not {r['path'] for r in meta['rows']}&{r['path'] for r in manifest['samples']}
  train_meta=json.loads((folder/'features/train.json').read_text());assert all(r['domain']=='id' for r in train_meta['rows'])
  ood=json.loads((folder/'ood_config.json').read_text());assert ood['calibration_split']=='val' and ood['fit_id_samples']==345 and ood['checkpoint_sha256']==training['checkpoint_sha256']
  assert sha256(folder/'ood/calibration.json')==ood['calibration_sha256']
  precision=np.asarray(ood['inverse_covariance']);assert precision.shape==(128,128) and np.linalg.eigvalsh(precision).min()>0
 assert all(paths==sets[0] for paths in sets)
 parity=read('results/v4_phase1/onnx_verification.json');browser=read('results/v4_phase1/browser_verification.json');artifact=read('artifacts/v4/model_config.json')
 assert parity['status']==browser['status']=='PASS' and parity['count']==544
 assert parity['max_logit_abs_difference']<1e-4 and parity['max_embedding_abs_difference']<1e-4
 assert parity['model_sha256']==sha256(ROOT/'artifacts/v4/model.onnx')==artifact['model_sha256']
 assert artifact['candidate']==selection['candidate']['name'] and artifact['production_replaced'] is False
 assert sha256(ROOT/'artifacts/v4/ood_config.json')==artifact['ood_config_sha256']
 assert browser['all_requests_static_local_get'] and browser['production_unchanged']
 for name in ['benchmark_audit','feedback_holdout_guard','benchmark_restore_verification']:assert read('results/v4_phase1/'+name+'.json')['status']=='PASS'
 assert read('results/v4_phase1/comparison.json')['production_replaced'] is False
 print('PASS: fixed321/121 holdout, same544 test rows, train/val-only model/calibration selection, exact metrics, actual ONNX/browser, preserved production')

if __name__=='__main__':main()
