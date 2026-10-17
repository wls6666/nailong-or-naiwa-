"""V4 held-out cartoon benchmark. No model imports/scoring; never used for fitting.
New Unicode identities (all old hard splits excluded), fixed published source versions.
"""
from pathlib import Path
import json,urllib.request,hashlib
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from PIL import Image
from dataset import ROOT,load_rgb,discover_splits,DEFAULT_DATA
from common import save_json,sha256
from prepare_hard_ood import SOURCES,pixel_hash

FACES=[0x1f910,0x1f911,0x1f912,0x1f913,0x1f914,0x1f915,0x1f917,*range(0x1f920,0x1f926),*range(0x1f927,0x1f930),*range(0x1f970,0x1f977),0x1f978,0x1f979,0x1f97a,0x1f9d0,*range(0x1fae0,0x1fae9),0x1f31a,0x1f31b,0x1f31c,0x1f47f,0x1f480,0x1f4a9,0x1f921]
ANIMALS=[0x1f43f,0x1f54a,0x1f577,0x1f578,*range(0x1f980,0x1f998),*range(0x1f9a0,0x1f9ae),*range(0x1fab0,0x1fabf),0x1fabf,0x1face,0x1fada,0x1facf,0x1fac3]
CHARACTERS=[0x1f466,0x1f467,0x1f468,0x1f469,0x1f474,0x1f475,0x1f476,0x1f471,0x1f385,0x1f478,0x1f47c,0x1f482,0x1f477,0x1f473,0x1f575,0x1f9d2,0x1f9d3,0x1f9d4,0x1f9d5,0x1f9d9,0x1f9da,0x1f9db,0x1f9dc,0x1f9dd,0x1f9de,0x1f9df,0x1f9cc,0x1f9b8,0x1f9b9]
NON_CHARACTER_CODES={0x1f578,0x1f9a0,0x1f9aa,*range(0x1fab4,0x1fabc),0x1fabd,0x1fabe,0x1fac3,0x1fada}
COMMIT='1ffb34c752ecf5d402f04cfb4b392c77f57c54bc'


def main():
 out=ROOT/'data/hard_ood_benchmark';out.mkdir(parents=True,exist_ok=True);manifest=ROOT/'data/hard_ood_test_manifest.json'
 reference=ROOT/'results/v4_phase1/hard_ood_test_manifest.json'
 if manifest.exists() or reference.exists():
  # Existing/published manifest defines the exact set; missing downloads are
  # restored, never silently omitted or replaced by newly available images.
  import time
  m=json.loads((manifest if manifest.exists() else reference).read_text())
  def restore(row):
   dest=ROOT/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    if sha256(dest)!=row['file_sha256']:raise ValueError(f'Existing file changed: {dest}')
    return
   for attempt in range(3):
    try:
     data=urllib.request.urlopen(row['url'],timeout=30).read()
     if hashlib.sha256(data).hexdigest()!=row['file_sha256']:raise ValueError('Source bytes changed')
     dest.write_bytes(data)
     if pixel_hash(dest)!=row['pixel_sha256']:raise ValueError('Decoded pixels changed')
     return
    except Exception:
     dest.unlink(missing_ok=True)
     if attempt==2:raise
     time.sleep(1)
  with ThreadPoolExecutor(max_workers=10) as pool:list(pool.map(restore,m['samples']))
  save_json(manifest,m)
  for source,info in SOURCES.items():
   lic=out/(source+'_LICENSE.txt')
   if not lic.exists():lic.write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/{info['repository']}/{info['commit']}/{info['license_file']}",timeout=30).read())
  lic=out/'fluent3d_LICENSE.txt'
  if not lic.exists():lic.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/microsoft/fluentui-emoji/{COMMIT}/LICENSE',timeout=30).read())
  print('Verified/restored exact locked benchmark',m['sample_count']);return
 old=json.loads((ROOT/'data/hard_ood_manifest.json').read_text());old_ids={r['identity_group'] for r in old['samples']}
 groups={f'{n:x}':cat for cat,nums in [('yellow_expressive_faces',FACES),('cartoon_animals_creatures',ANIMALS),('cute_humanoid_fantasy_characters',CHARACTERS)] for n in nums if f'{n:x}' not in old_ids and n not in NON_CHARACTER_CODES}
 # Exact RGB duplicates against ALL original train/val/test, old hard and feedback.
 reserved=set()
 for folder in discover_splits(DEFAULT_DATA).values():
  for p in folder.rglob('*'):
   if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:reserved.add(pixel_hash(p))
 reserved.update(r['pixel_sha256'] for r in old['samples'])
 for folder in ['hard_negative','false_positive']:
  for p in (ROOT/'data'/folder).rglob('*'):
   if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:reserved.add(pixel_hash(p))
 base=f'https://raw.githubusercontent.com/microsoft/fluentui-emoji/{COMMIT}/'
 tree=json.load(urllib.request.urlopen(f'https://api.github.com/repos/microsoft/fluentui-emoji/git/trees/{COMMIT}?recursive=1',timeout=60));assert not tree.get('truncated')
 # Filter semantic folder names only; metadata verifies identity. No inference is used.
 folders=[r['path'] for r in tree['tree'] if r['path'].endswith('/metadata.json')]
 words=['face','cat','dog','animal','frog','dragon','dinosaur','sauropod','bird','bear','teddy','robot','ghost','alien','monkey','cow','pig','rabbit','mouse','rat','unicorn','tiger','lion','fox','wolf','hamster','panda','koala','duck','chicken','smil','grinn','laugh','smirk','worr','angry','cry','weary','sleep','kiss','wink','melt','neutral','expressionless','moon','sun','shrug','boy','girl','child','baby','person','man','woman','elf','fairy','vampire','zombie','mage','genie','mermaid','super','goblin','troll','lion','deer','bat','boar','squirrel','eagle','owl','swan','parrot','peacock','dodo','flamingo','beaver','bison','sloth','otter','orangutan','seal','mammoth','llama','kangaroo','hedgehog','raccoon','badger','hippopotamus','lobster','butterfly','snail','worm','mosquito','beetle','cricket','scorpion','skunk','turkey','gorilla','fox','ostrich','goose','donkey','moose','jellyfish','shrimp','squid','crab']
 folders=[p for p in folders if any(w in p.split('/')[1].lower() for w in words)]
 def meta(path):
  try:
   m=json.load(urllib.request.urlopen(base+quote(path),timeout=20));code=m['unicode'].lower()
   if code in groups:return code,path.rsplit('/',1)[0]
  except Exception:return None
 with ThreadPoolExecutor(max_workers=10) as pool:fluent=[r for r in pool.map(meta,folders) if r]
 paths={r['path'] for r in tree['tree']}
 jobs=[]
 for code,category in sorted(groups.items()):
  for source,s in SOURCES.items():jobs.append((source,code,category,f"https://raw.githubusercontent.com/{s['repository']}/{s['commit']}/"+s['asset'].format(code=code),s['license']))
 for code,folder in fluent:
  candidates=sorted(p for p in paths if p.startswith(folder+'/3D/') and p.endswith('_3d.png'))
  if candidates:jobs.append(('fluent3d',code,groups[code],base+quote(candidates[0]),'MIT'))
 def fetch(job):
  source,code,category,url,license=job;dest=out/f'{source}_{code}.png'
  try:
   dest.write_bytes(urllib.request.urlopen(url,timeout=25).read())
   with Image.open(dest) as im:im.verify()
   return {'path':str(dest.relative_to(ROOT)),'source':source,'source_id':code,'identity_group':code,'category':category,'hard_ood':True,'used_for_training':False,'used_for_validation':False,'split':'test_only','url':url,'license':license,'file_sha256':sha256(dest),'pixel_sha256':pixel_hash(dest)}
  except Exception as e:
   dest.unlink(missing_ok=True);return {'missing':source+'_'+code,'error':str(e)}
 with ThreadPoolExecutor(max_workers=10) as pool:got=list(pool.map(fetch,jobs))
 samples=[];excluded=[];seen=set()
 for row in got:
  if 'missing' in row:excluded.append(row);continue
  h=row['pixel_sha256']
  if h in reserved or h in seen:
   excluded.append({'path':row['path'],'reason':'duplicate/reserved pixels'});(ROOT/row['path']).unlink();continue
  assert row['identity_group'] not in old_ids
  seen.add(h);samples.append(row)
 assert len(samples)>=100,('Insufficient real benchmark images',len(samples))
 m={'version':'v4_phase1_cartoon_holdout_1','sample_count':len(samples),'identity_count':len({r['identity_group'] for r in samples}),'samples':samples,'excluded_or_unavailable':excluded,'source_versions':{**SOURCES,'fluent3d':{'repository':'microsoft/fluentui-emoji','commit':COMMIT,'license':'MIT'}},'counts_by_source':dict(Counter(r['source'] for r in samples)),'counts_by_category':dict(Counter(r['category'] for r in samples)),'old_v3_identity_overlap':0,'project_cross_split_exact_pixel_overlap':0,'locked_before_evaluation':True,'protocol_sha256':sha256(ROOT/'experiments/v4_phase1/protocol.json'),'limitations':['Cartoon icon/3D rendered art; no broad plush photograph or commercial franchise coverage.','New Unicode identity groups, but sources/styles shared with old exposure; close facial variants can remain.','No image-level audit of historical ImageNet pretraining corpus.','All instances test-only: forbidden to train/calibrate/tune on this set.']}
 for source,s in SOURCES.items():
  (out/(source+'_LICENSE.txt')).write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/{s['repository']}/{s['commit']}/{s['license_file']}",timeout=25).read())
 (out/'fluent3d_LICENSE.txt').write_bytes(urllib.request.urlopen(base+'LICENSE',timeout=25).read())
 save_json(manifest,m);save_json(ROOT/'results/v4_phase1/hard_ood_test_manifest.json',m)
 save_json(ROOT/'results/v4_phase1/hard_ood_manifest_summary.json',{k:v for k,v in m.items() if k not in ['samples','excluded_or_unavailable']})
 print('LOCKED benchmark',len(samples),'identities',m['identity_count'],m['counts_by_source'],m['counts_by_category'],flush=True)

if __name__=='__main__':main()
