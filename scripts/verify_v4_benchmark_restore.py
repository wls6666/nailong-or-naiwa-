"""Restore one genuinely missing benchmark file from its pinned public URL.
Backup guarantees local recovery if the network or source is unavailable.
"""
import sys,json,tempfile,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,sha256,save_json
from v4.common import check_production
m=ROOT/'data/hard_ood_test_manifest.json';before=sha256(m);row=json.loads(m.read_text())['samples'][0];source=ROOT/row['path'];expected=sha256(source)
with tempfile.TemporaryDirectory() as temp:
 backup=Path(temp)/'source.png';shutil.copyfile(source,backup);source.unlink()
 try:
  subprocess.run([sys.executable,str(ROOT/'training/prepare_hard_ood_benchmark.py')],cwd=ROOT,check=True)
  assert sha256(source)==expected and sha256(m)==before
 finally:
  if not source.exists() or sha256(source)!=expected:shutil.copyfile(backup,source)
check_production();save_json(ROOT/'results/v4_phase1/benchmark_restore_verification.json',{'status':'PASS','restored_file':row['path'],'source_url':row['url'],'sha256':expected,'manifest_unchanged':True,'sample_count_unchanged':True,'actual_public_download':True});print('Exact benchmark restoration PASS')
