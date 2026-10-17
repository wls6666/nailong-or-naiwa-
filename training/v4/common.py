"""Isolated V4 data/model utilities. Production V3 files are read-only here."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from torch.utils.data import Dataset,DataLoader
from dataset import ROOT,DEFAULT_DATA,discover_splits,load_rgb
from common import save_json,sha256

BASE=ROOT/'results/v4_phase1'


def resize(image,size):
 pixels=np.asarray(image,dtype=np.float32);h,w=pixels.shape[:2]
 x=np.maximum(0,(np.arange(size,dtype=np.float64)+.5)*w/size-.5);y=np.maximum(0,(np.arange(size,dtype=np.float64)+.5)*h/size-.5)
 x0,y0=np.floor(x).astype(int),np.floor(y).astype(int);x1,y1=np.minimum(x0+1,w-1),np.minimum(y0+1,h-1)
 wx,wy=(x-x0)[None,:,None],(y-y0)[:,None,None]
 top=pixels[y0[:,None],x0]*(1-wx)+pixels[y0[:,None],x1]*wx;bottom=pixels[y1[:,None],x0]*(1-wx)+pixels[y1[:,None],x1]*wx
 return torch.from_numpy(np.ascontiguousarray(((top*(1-wy)+bottom*wy)/255).transpose(2,0,1),dtype=np.float32))


def rows(split):
 ds=ImageFolder(str(discover_splits(DEFAULT_DATA)[split]));assert ds.class_to_idx=={'nailong':0,'naiwa':1,'other':2},ds.class_to_idx
 result=[{'path':str(Path(p).relative_to(ROOT)),'label':int(y),'class':ds.classes[y],'domain':'easy' if y==2 else 'id','split':split} for p,y in ds.samples if split!='train' or y!=2]
 if split=='val':
  old=json.loads((ROOT/'data/hard_ood_manifest.json').read_text());result.extend({**r,'label':2,'class':'other','domain':'hard'} for r in old['samples'] if r['split']=='val')
 if split=='test':
  manifest=json.loads((ROOT/'data/hard_ood_test_manifest.json').read_text());result.extend({**r,'label':2,'class':'other','domain':'hard'} for r in manifest['samples'])
 return result


def set_hash(items):
 return hashlib.sha256(json.dumps([{'path':r['path'],'sha256':sha256(ROOT/r['path']),'label':r['label'],'domain':r['domain']} for r in items],sort_keys=True).encode()).hexdigest()


class Images(Dataset):
 def __init__(self,items,size,augment=False):
  self.rows=items;self.cache=[resize(load_rgb(ROOT/r['path']),size) for r in items]
  self.augment=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(8,interpolation=transforms.InterpolationMode.BILINEAR,fill=1),transforms.ColorJitter(brightness=.12,contrast=.12,saturation=.1)]) if augment else None
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return (self.augment(self.cache[i]) if self.augment else self.cache[i]),self.rows[i]['label']


class MobileNet(nn.Module):
 def __init__(self,pretrained=False):
  super().__init__();self.features=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None).features
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
  self.projection=nn.Linear(576,128);self.dropout=nn.Dropout(.2);self.classifier=nn.Linear(128,2)
 def pooled(self,x):return self.features((x-self.mean)/self.std).mean((2,3))
 def head(self,x):
  embedding=self.projection(x);return self.classifier(self.dropout(embedding)),embedding
 def forward_outputs(self,x):return self.head(self.pooled(x))
 def forward(self,x):return self.forward_outputs(x)[0]


def load(name):
 if name=='v3_baseline':
  from common import load_model
  model,saved=load_model(ROOT/'checkpoints/best_model_ood.pth');return model,128,ROOT/'checkpoints/best_model_ood.pth'
 path=ROOT/f'checkpoints/v4_phase1/{name}.pth';saved=torch.load(path,map_location='cpu',weights_only=True)
 model=MobileNet();model.load_state_dict(saved['model_state_dict']);model.eval();return model,saved['input_size'],path


def features(model,ds,pooled=False):
 arrays=[];logits=[]
 with torch.inference_mode():
  for x,y in DataLoader(ds,batch_size=32,shuffle=False):
   if pooled:arrays.extend(model.pooled(x).numpy())
   else:
    l,e=model.forward_outputs(x);logits.extend(l.numpy());arrays.extend(e.numpy())
 return np.asarray(arrays,dtype=np.float32) if pooled else (np.asarray(logits,dtype=np.float32),np.asarray(arrays,dtype=np.float32))


def extract(name,splits):
 model,size,path=load(name);model.eval();folder=BASE/name/'features';folder.mkdir(parents=True,exist_ok=True)
 for split in splits:
  items=rows(split);ds=Images(items,size);logits,embedding=features(model,ds)
  np.savez_compressed(folder/f'{split}.npz',logits=logits,embedding=embedding,labels=np.asarray([r['label'] for r in items]))
  save_json(folder/f'{split}.json',{'rows':items,'class_to_idx':{'nailong':0,'naiwa':1},'checkpoint_sha256':sha256(path),'embedding_dim':embedding.shape[1],'input_size':size,'set_sha256':set_hash(items)})
  print(name,split,len(items),logits.shape,embedding.shape,flush=True)
 return folder


def check_production():
 frozen=json.loads((ROOT/'results/v3_baseline/freeze.json').read_text())
 assert {str(p.relative_to(ROOT)) for p in (ROOT/'web').rglob('*') if p.is_file()}==set(frozen['production_web_sha256']), 'Production file set changed'
 for path,digest in frozen['production_web_sha256'].items():assert sha256(ROOT/path)==digest,('Production changed',path)
 assert sha256(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
