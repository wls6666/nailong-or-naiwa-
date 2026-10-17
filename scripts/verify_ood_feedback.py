"""Verify that a real reserved hard-test image cannot enter feedback training.
Temporary copy removed even on error.
"""
import sys,shutil,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from dataset import ROOT,datasets
from common import save_json
source=ROOT/'data/hard_ood/test/fluent3d_1f438.png';dest=ROOT/'data/false_positive/.v3_holdout_probe.png'
assert not dest.exists();shutil.copyfile(source,dest)
try:
 try:datasets(training=True)
 except ValueError as error:
  assert 'leaks validation/test' in str(error);message=str(error)
 else:raise AssertionError('Holdout was admitted to training')
finally:dest.unlink()
save_json(ROOT/'results/v3/feedback_verification.json',{'status':'PASS','hard_test_feedback_rejected':True,'source':str(source.relative_to(ROOT)),'message':message,'temporary_file_removed':not dest.exists(),'new_feedback_loader_insertion':'Previously exercised by results/feedback_verification.json; same insertion mechanism retained. Hard test reservation additionally verified here.'})
print('Hard OOD holdout rejected from feedback PASS')
