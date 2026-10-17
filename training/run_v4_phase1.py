"""Reproducible V4 Phase 1. Isolated outputs; V3 production is never modified.
Use --resume only to reuse checkpoints whose hashes and protocol match.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MPLCONFIGDIR','/tmp/naiwa-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,save_json,sha256
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/torch_cache'))
import torch
from v4.common import BASE,extract,rows,set_hash,check_production
from v4.detector import calibrate,key
from v4.evaluate import evaluate


def run(script,log,*args):
 print('RUN',script,*args,flush=True)
 with Path(log).open('w') as stream:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)


def main():
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--browser',action='store_true',help='Also run local real Chromium/WASM verification');args=p.parse_args();torch.set_num_threads(2);BASE.mkdir(parents=True,exist_ok=True);check_production()
 run('training/prepare_hard_ood_benchmark.py',BASE/'prepare_benchmark_final.log')
 protocol=ROOT/'experiments/v4_phase1/protocol.json';benchmark=ROOT/'data/hard_ood_test_manifest.json'
 lock={'protocol_sha256':sha256(protocol),'benchmark_sha256':sha256(benchmark),'train_set_sha256':set_hash(rows('train')),'val_set_sha256':set_hash(rows('val')),'test_set_sha256':set_hash(rows('test')),'test_set_use':'hash/audit only before final evaluation; never fitting or hyperparameter selection'}
 lockfile=ROOT/'experiments/v4_phase1/data_lock.json'
 if lockfile.exists():assert json.loads(lockfile.read_text())==lock,'Data/protocol changed; refuse silent reuse'
 else:save_json(lockfile,lock)
 if not args.resume or not (BASE/'v3_baseline/metrics.json').exists():run('training/v4/baseline.py',BASE/'v3_baseline.log')
 names=['mobilenet_frozen','mobilenet_partial_ft','mobilenet_frozen_128_control'];experiments=[]
 for name in names:
  folder=BASE/name;folder.mkdir(exist_ok=True);summary=folder/'training_summary.json'
  if args.resume and summary.exists():
   saved=json.loads(summary.read_text());assert sha256(ROOT/saved['checkpoint'])==saved['checkpoint_sha256'];assert json.loads((folder/'training_config.json').read_text())['protocol_sha256']==lock['protocol_sha256']
   print('RESUME verified training',name,flush=True)
  else:run('training/v4/train.py',folder/'training.log',name)
  extract(name,['train','val']);cal=calibrate(name);rec={'name':name,'validation':cal['selected'],'checkpoint_sha256':cal['checkpoint_sha256'],'calibration_sha256':sha256(folder/'ood/calibration.json')}
  rec['eligible_id']=rec['validation']['closed_set_id_accuracy']>=.9178082191780822-.03;experiments.append(rec)
 eligible=[r for r in experiments if r['eligible_id']];candidate=min(eligible or experiments,key=lambda r:key(r['validation']))
 selection={'selection_split':'val','test_used':False,'data_lock':lock,'experiments':experiments,'candidate':candidate,'note':'Only this candidate may be exported. No adjustment after test. Eligibility controls recommendation, not test reporting.'}
 save_json(BASE/'candidate_selection.json',selection);print('FROZEN MOBILE CANDIDATE',candidate['name'],flush=True)
 # The new hard benchmark is first passed to MobileNet models only after all choices above.
 for name in names:extract(name,['test']);evaluate(name)
 from v4.report import report
 report();check_production()
 run('training/v4/export.py',BASE/'onnx_export.log')
 if args.browser:run('scripts/verify_v4_browser.py',BASE/'browser_verification.log')
 print('V4 PHASE1 COMPLETE; production unchanged',flush=True)

if __name__=='__main__':main()
