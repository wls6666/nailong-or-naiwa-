"""Independent membership/pixel/identity audit for permanent V4 held-out test."""
import sys,json
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,sha256,save_json
from prepare_hard_ood import pixel_hash
from dataset import discover_splits,DEFAULT_DATA
from v4.common import check_production


def main():
 m=json.loads((ROOT/'data/hard_ood_test_manifest.json').read_text());old=json.loads((ROOT/'data/hard_ood_manifest.json').read_text());old_ids={r['identity_group'] for r in old['samples']}
 reserved={r['pixel_sha256'] for r in old['samples']}
 for folder in discover_splits(DEFAULT_DATA).values():
  for p in folder.rglob('*'):
   if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:reserved.add(pixel_hash(p))
 for directory in ['hard_negative','false_positive']:
  for p in (ROOT/'data'/directory).rglob('*'):
   if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:reserved.add(pixel_hash(p))
 seen=set()
 for r in m['samples']:
  assert r['split']=='test_only' and r['hard_ood'] and not r['used_for_training'] and not r['used_for_validation']
  assert r['identity_group'] not in old_ids
  assert sha256(ROOT/r['path'])==r['file_sha256'];h=pixel_hash(ROOT/r['path']);assert h==r['pixel_sha256']
  assert h not in reserved and h not in seen;seen.add(h)
 assert len(seen)==m['sample_count']
 check_production();save_json(ROOT/'results/v4_phase1/benchmark_audit.json',{'status':'PASS','count':m['sample_count'],'identities':m['identity_count'],'old_v3_identity_overlap':0,'project_reserved_pixel_overlap':0,'within_benchmark_pixel_duplicates':0,'manifest_sha256':sha256(ROOT/'data/hard_ood_test_manifest.json'),'all_test_only':True,'production_unchanged':True,'limitations':m['limitations']});print('Independent benchmark audit PASS',m['sample_count'])

if __name__=='__main__':main()
