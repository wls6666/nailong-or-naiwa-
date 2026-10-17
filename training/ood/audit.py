"""Audit decoded pixels and identity groups without running models or calibrating.
Run after both data preparation scripts.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import DEFAULT_DATA,discover_splits,CharacterDataset,ROOT
from common import save_json
from prepare_hard_ood import pixel_hash
manifest=json.loads((ROOT/'data/hard_ood_manifest.json').read_text())
seen={};identities={};counts={};duplicates=[]
for split,folder in discover_splits(DEFAULT_DATA).items():
 ds=CharacterDataset(folder);counts[split]={'original':len(ds),'hard':0}
 for path,label in ds.samples:
  h=pixel_hash(path)
  if h in seen and seen[h]['split']!=split:duplicates.append([seen[h],{'split':split,'path':str(path)}])
  seen[h]={'split':split,'path':str(path)}
for row in manifest['samples']:
 h=pixel_hash(ROOT/row['path']);assert h==row['pixel_sha256']
 if h in seen and seen[h]['split']!=row['split']:duplicates.append([seen[h],row])
 seen[h]=row
 ident=row['identity_group'];assert identities.get(ident,row['split'])==row['split'];identities[ident]=row['split'];counts[row['split']]['hard']+=1
assert not duplicates,duplicates
save_json(ROOT/'results/v3/data_audit.json',{'status':'PASS','counts':counts,'hard_identity_groups':len(identities),'cross_split_pixel_collisions':duplicates,'pixel_hash':'SHA256(RGB dimensions + decoded RGB bytes; white alpha)','original_splits_preserved':True,'feedback_test_reservation':'training/dataset.py excludes hard OOD val/test hashes from hard_negative/false_positive training additions','limitation':'Exact pixel/identity grouping does not prove absence of similar distinct emoji; sources share visual style across splits.'})
print(counts,'PASS')
