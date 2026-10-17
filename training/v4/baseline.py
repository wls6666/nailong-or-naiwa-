"""Re-run actual frozen V3 on fixed new benchmark; no detector adjustment."""
import sys,json,shutil
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save_json,sha256
from v4.common import BASE,extract,check_production
from v4.evaluate import evaluate

def main():
 torch.set_num_threads(2);check_production();folder=BASE/'v3_baseline';folder.mkdir(parents=True,exist_ok=True);(folder/'ood').mkdir(exist_ok=True)
 extract('v3_baseline',['train','val','test'])
 shutil.copyfile(ROOT/'web/ood_config.json',folder/'ood_config.json')
 for f in ['calibration.json','gaussian_params.json']:shutil.copyfile(ROOT/'results/v3/phase_b'/f,folder/'ood'/f)
 stats=evaluate('v3_baseline');save_json(ROOT/'results/v3_baseline/v4_benchmark_metrics.json',stats);check_production()

if __name__=='__main__':main()
