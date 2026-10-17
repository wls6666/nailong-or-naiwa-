"""Small licensed cartoon-icon OOD probe, grouped by Unicode identity across sources.
Not a comprehensive benchmark of 3D cartoons, plush toys or user error cases.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import urllib.request
from urllib.parse import quote
from PIL import Image
from dataset import ROOT,load_rgb
from common import save_json,sha256

SOURCES={
 'noto': {'commit':'v2.047','repository':'googlefonts/noto-emoji','asset':'png/128/emoji_u{code}.png','license':'Apache-2.0','license_file':'LICENSE'},
 'twemoji': {'commit':'7407fa31c51be5ab45626b8ab5554d50cc8073f6','repository':'jdecked/twemoji','asset':'assets/72x72/{code}.png','license':'CC-BY-4.0','license_file':'LICENSE-GRAPHICS'},
}
GROUPS={
 'yellow_faces': [f'{x:x}' for x in range(0x1f600,0x1f638)],
 'cartoon_animals': [f'{x:x}' for x in range(0x1f400,0x1f43f)],
 'creatures_toys': ['1f479','1f47a','1f47b','1f47d','1f47e','1f916','1f9f8','1f984','1f995','1f996','1f31d','1f31e'],
}


def pixel_hash(p):
 im=load_rgb(p);return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()


def main():
 target=ROOT/'data/hard_ood';target.mkdir(parents=True,exist_ok=True)
 manifest_path=ROOT/'data/hard_ood_manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  for row in manifest['samples']:assert sha256(ROOT/row['path'])==row['file_sha256']
  if 'fluent3d' not in manifest['sources']:manifest=add_fluent_3d(manifest)
  save_json(ROOT/'results/v3/hard_ood_manifest.json',manifest)
  print('Verified existing hard OOD manifest',manifest['counts']);return
 assignments={};rng=random.Random(42)
 for category,codes in GROUPS.items():
  codes=sorted(set(codes));rng.shuffle(codes)
  for i,code in enumerate(codes):
   # Identity grouping: no emoji has another source rendering in a different split.
   split='train' if i%5<2 else ('val' if i%5==2 else 'test')
   assignments[code]=(category,split)
 # These independent identity groups are held out, never training instances.
 for code in ['1f438','1f432','1f995','1f425']:assignments[code]=(assignments[code][0],'test')
 jobs=[(source,code,category,split) for code,(category,split) in sorted(assignments.items()) for source in SOURCES]
 def fetch(job):
  source,code,category,split=job;s=SOURCES[source]
  url=f"https://raw.githubusercontent.com/{s['repository']}/{s['commit']}/"+s['asset'].format(code=code)
  path=target/split/(source+'_'+code+'.png');path.parent.mkdir(exist_ok=True)
  try:
   data=urllib.request.urlopen(url,timeout=30).read();path.write_bytes(data)
   with Image.open(path) as im:im.verify()
   return {'source':source,'source_id':code,'identity_group':code,'category':category,'split':split,
           'path':str(path.relative_to(ROOT)),'url':url,'license':s['license'],
           'file_sha256':sha256(path),'pixel_sha256':pixel_hash(path)}
  except Exception as e:
   if path.exists():path.unlink()
   return {'missing':True,'source':source,'source_id':code,'error':str(e)}
 with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(fetch,jobs))
 samples=[];missing=[];seen={}
 for row in results:
  if row.get('missing'):missing.append(row);continue
  digest=row['pixel_sha256']
  if digest in seen:
   # Exact duplicates are omitted, never allowed into another split.
   missing.append({'source':row['source'],'source_id':row['source_id'],'error':'exact duplicate pixels'});continue
  seen[digest]=row['split'];samples.append(row)
 for source,s in SOURCES.items():
  license_url=f"https://raw.githubusercontent.com/{s['repository']}/{s['commit']}/{s['license_file']}"
  (target/(source+'_LICENSE.txt')).write_bytes(urllib.request.urlopen(license_url,timeout=30).read())
 identities={}
 for row in samples:
  assert row['identity_group'] not in identities or identities[row['identity_group']]==row['split']
  identities[row['identity_group']]=row['split']
 counts={split:sum(r['split']==split for r in samples) for split in ['train','val','test']}
 assert counts['val']>=20 and counts['test']>=30,counts
 manifest={'seed':42,'sources':SOURCES,'samples':samples,'missing':missing,'counts':counts,
           'scope':'Curated yellow cartoon faces, cartoon animals, dragons/frog/dinosaurs and toy/creature icons. Does not cover realistic 3D cartoons, plush photographs or all decision-boundary cases.',
           'leakage':'Same Unicode identity in both sources is assigned to one split; exact pixels unique. Similar distinct emoji may remain.',
           'phase_a_train_usage':'No hard train image used to fit the encoder or class centers; reserved for optional Phase B.'}
 manifest=add_fluent_3d(manifest)
 save_json(manifest_path,manifest);save_json(ROOT/'results/v3/hard_ood_manifest.json',manifest)
 print('Hard OOD prepared',counts,'missing',len(missing),flush=True)


def add_fluent_3d(manifest):
 m=manifest;groups={r['source_id']:(r['category'],r['split']) for r in m['samples']}
 commit='1ffb34c752ecf5d402f04cfb4b392c77f57c54bc'
 tree=json.load(urllib.request.urlopen(f'https://api.github.com/repos/microsoft/fluentui-emoji/git/trees/{commit}?recursive=1',timeout=60))
 assert not tree.get('truncated')
 paths=[r['path'] for r in tree['tree'] if r['path'].endswith('_3d.png') and '/3D/' in r['path'] and len(r['path'].split('/'))==4]
 # Cartoon faces, animals and creature characters; metadata provides identity, never filename guessing.
 keys=['face','cat','dog','frog','dragon','dinosaur','t-rex','sauropod','chick','bird','bear','teddy','robot','ghost','alien','monkey','cow','pig','rabbit','mouse','rat','unicorn','tiger','lion','fox','wolf','hamster','panda','koala','duck','chicken','ogre','goblin','smiling','grinning','laughing','smirk','worried','angry','crying','weary','yawning','sleep','kiss','wink','melting','neutral','expressionless','smile','moon','sun']
 paths=[p for p in paths if any(k in p.split('/')[1].lower() for k in keys)]
 print('Candidate 3D character folders',len(paths),flush=True)
 base=f'https://raw.githubusercontent.com/microsoft/fluentui-emoji/{commit}/'
 def get(path):
  try:
   folder=path.rsplit('/3D/',1)[0];meta=json.load(urllib.request.urlopen(base+quote(folder+'/metadata.json'),timeout=25));code=meta['unicode'].lower()
   if code not in groups:return None
   category,split=groups[code];dest=ROOT/f'data/hard_ood/{split}/fluent3d_{code}.png';dest.write_bytes(urllib.request.urlopen(base+quote(path),timeout=25).read())
   return {'source':'fluent3d','source_id':code,'identity_group':code,'category':category,'split':split,'path':str(dest.relative_to(ROOT)),'url':base+quote(path),'license':'MIT','file_sha256':sha256(dest),'pixel_sha256':pixel_hash(dest)}
  except Exception as e:print('failed',path,type(e).__name__,flush=True);return None
 with ThreadPoolExecutor(max_workers=8) as pool:rows=[r for r in pool.map(get,paths) if r]
 seen={r['pixel_sha256'] for r in m['samples']}
 for row in rows:
  if row['pixel_sha256'] not in seen:m['samples'].append(row);seen.add(row['pixel_sha256'])
 m['sources']['fluent3d']={'commit':commit,'repository':'microsoft/fluentui-emoji','license':'MIT','style':'3D rendered cartoon emoji','selection':'existing Unicode identity groups only; no score-dependent selection'}
 m['counts']={s:sum(r['split']==s for r in m['samples']) for s in ['train','val','test']}
 m['scope']='Licensed 2D + 3D cartoon emoji faces, animals, dragons/frogs/dinosaurs/creatures. Not a representative benchmark of plush photographs or third-party commercial cartoon characters.'
 (ROOT/'data/hard_ood/fluent3d_LICENSE.txt').write_bytes(urllib.request.urlopen(base+'LICENSE',timeout=20).read())
 save_json(ROOT/'data/hard_ood_manifest.json',m);save_json(ROOT/'results/v3/hard_ood_manifest.json',m)
 print('Added',len(rows),'Fluent 3D; counts',m['counts'],flush=True)
 return m


if __name__=='__main__':main()
