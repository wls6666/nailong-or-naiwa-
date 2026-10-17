"""Verify binding of actual V3 weights/model/calibration/evaluations, and V2 archive."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,sha256,load_model

def read(path):return json.loads((ROOT/path).read_text())
def main():
 selection=read('results/v3/production_selection.json');chosen=selection['selected'];assert selection['selection_split']=='val' and selection['test_used'] is False
 config=read('web/model_config.json');ood=read('web/ood_config.json');cal=read(chosen['directory']+'/calibration.json');gp=read(chosen['directory']+'/gaussian_params.json')
 assert sha256(ROOT/chosen['checkpoint'])==chosen['checkpoint_sha256']==ood['checkpoint_sha256']==config['checkpoint_sha256']==cal['checkpoint_sha256']
 assert sha256(ROOT/'web/model.onnx')==ood['model_sha256']==config['model_sha256']
 assert sha256(ROOT/'web/ood_config.json')==config['ood_config_sha256']
 assert sha256(ROOT/chosen['directory']/'calibration.json')==ood['calibration_sha256']
 assert sha256(ROOT/chosen['directory']/'gaussian_params.json')==cal['gaussian_params_sha256']
 assert cal['calibration_split']=='val' and chosen['validation']['parameters']==cal['selected']['parameters']
 assert ood['mahalanobis_threshold']==cal['selected']['parameters']['distance_threshold']
 assert ood['energy_threshold']==cal['selected']['parameters']['energy_threshold']
 assert ood['normalize_embedding']==gp['normalize_embedding'] and ood['fit_id_samples']==345
 assert gp['class_centers']==ood['class_centers'] and gp['inverse_covariance']==ood['inverse_covariance']
 precision=np.asarray(ood['inverse_covariance']);assert precision.shape==(256,256) and np.isfinite(precision).all() and np.linalg.eigvalsh(precision).min()>0
 assert config['logits_shape']==[1,2] and config['embedding_shape']==[1,256] and config['output_names']==['logits','embedding']
 model,saved=load_model(ROOT/chosen['checkpoint']);assert saved['class_to_idx']==config['class_to_idx']
 history=read('results/v3/oe_training_history.json');tc=read('results/v3/oe_training_config.json')
 best=max(history,key=lambda r:(r['val_id_accept']>=.95 and r['val_id_accuracy']>=tc['baseline_val_id_accuracy']-tc['maximum_val_id_accuracy_drop'],-max(r['val_easy_far'],r['val_hard_far']),-(r['val_easy_far']+r['val_hard_far'])/2,r['val_id_end_to_end_accuracy'],r['val_id_accuracy']))
 if chosen['family']=='custom_energy_oe':assert saved['epoch']==best['epoch']
 verification=read('results/v3/onnx_verification.json');browser=read('results/v3/browser_verification.json')
 assert verification['status']==browser['status']=='PASS' and verification['count']==372
 assert verification['max_logit_difference']<1e-4 and verification['max_embedding_difference']<1e-4
 assert verification['model_sha256']==browser['model_sha256']==config['model_sha256'] and browser['ood_config_sha256']==config['ood_config_sha256']
 assert browser['all_requests_local_get'] and not browser['page_errors']
 scores=list(csv.DictReader((ROOT/'results/v3/ood_scores.csv').open()));rows=[r for r in scores if r['split']=='test'];metrics=read('results/v3/metrics.json')['selected']
 assert len(rows)==372
 for domain in ['easy','hard']:
  subset=[r for r in rows if r['true_type']==domain];far=sum(r['accepted']=='True' for r in subset)/len(subset)
  assert len(subset)==metrics[domain+'_ood_count'] and abs(far-metrics[domain+'_ood_far'])<1e-12
 ids=[r for r in rows if r['true_type']=='id'];assert len(ids)==73
 assert abs(sum(r['accepted']=='True' for r in ids)/73-metrics['id_true_accept_rate'])<1e-12
 assert abs(sum(r['correct']=='True' for r in ids)/73-metrics['id_end_to_end_accuracy'])<1e-12
 v2=read('results/v2/onnx_verification_3class.json');assert sha256(ROOT/'checkpoints/v2/model.onnx')==v2['model_sha256'] and sha256(ROOT/'checkpoints/v2/best_model.pth')==v2['checkpoint_sha256']
 assert sha256(ROOT/'checkpoints/best_model_3class.pth')==v2['checkpoint_sha256']
 assert read('results/v3/data_audit.json')['status']=='PASS'
 for file,digest in read('web/assets/vendor/onnxruntime-web/version.json')['sha256'].items():assert sha256(ROOT/'web/assets/vendor/onnxruntime-web'/file)==digest
 print('PASS: V3 checkpoint, two-output ONNX, calibration hashes, positive covariance, exact score-derived metrics, real browser and preserved V2')

if __name__=='__main__':main()
