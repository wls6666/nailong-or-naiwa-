"""Ensure a real new benchmark image cannot enter the existing OE feedback loader."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from common import ROOT,save_json
from dataset import datasets
from v4.common import check_production
m=json.loads((ROOT/'data/hard_ood_test_manifest.json').read_text());source=ROOT/m['samples'][0]['path'];dest=ROOT/'data/false_positive/.v4_holdout_guard_probe.png'
assert not dest.exists();shutil.copyfile(source,dest)
try:
 try:datasets(training=True)
 except ValueError as e:
  assert 'leaks validation/test' in str(e);message=str(e)
 else:raise AssertionError('New benchmark leaked into feedback train')
finally:dest.unlink()
check_production();save_json(ROOT/'results/v4_phase1/feedback_holdout_guard.json',{'status':'PASS','test_image':str(source.relative_to(ROOT)),'error':message,'temporary_file_removed':True,'production_unchanged':True});print('V4 permanent benchmark feedback guard PASS')
