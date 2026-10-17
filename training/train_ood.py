"""Phase B: two ID logits + simple energy-margin OE; never cluster all Other.
Encoder initialized from this project's V2 CNN. No ImageNet model/SupCon.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from dataset import ROOT,DEFAULT_DATA,discover_splits,CharacterDataset,load_rgb,resize_rgb
from common import load_model,seed_all,save_json,sha256
from model import SmallCNN
from ood.core import fit_gaussian,distance,energy


class Cache(Dataset):
 def __init__(self,rows,augment=None):
  self.rows=rows;self.cache=[resize_rgb(load_rgb(ROOT/r['path'])) for r in rows];self.augment=augment
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x=self.cache[i]
  return self.augment(x) if self.augment else x


def outputs(model,ds):
 logits=[];emb=[]
 with torch.inference_mode():
  for batch in DataLoader(ds,batch_size=32):
   x=batch[0] if isinstance(batch,(tuple,list)) else batch
   l,e=model.forward_outputs(x);logits.extend(l.numpy());emb.extend(e.numpy())
 return np.asarray(logits),np.asarray(emb)


def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--oe-weight',type=float,default=.1);args=p.parse_args()
 torch.set_num_threads(2);seed_all(42)
 old,saved=load_model(ROOT/'checkpoints/v2/best_model.pth');model=SmallCNN(num_classes=2)
 model.features.load_state_dict(old.features.state_dict())
 target_indices=[saved['class_to_idx'][n] for n in ['nailong','naiwa']]
 with torch.no_grad():
  model.head[3].weight.copy_(old.head[3].weight[target_indices]);model.head[3].bias.copy_(old.head[3].bias[target_indices])
 splits=discover_splits(DEFAULT_DATA);train=CharacterDataset(splits['train'],augment=True);val=CharacterDataset(splits['val'])
 if train.class_to_idx!=saved['class_to_idx'] or val.class_to_idx!=train.class_to_idx:raise ValueError('Dataset/checkpoint mappings differ')
 id_mapping={name:train.class_to_idx[name] for name in ['nailong','naiwa']}
 if list(id_mapping.values())!=[0,1]:raise ValueError('Two-head initialization requires explicit target reindexing for this dataset mapping')
 # Original target identities/splits never move; all Other serves OE, not a third CE label.
 ids=[i for i,(_,y) in enumerate(train.samples) if y in target_indices]
 id_ds=torch.utils.data.Subset(train,ids)
 easy=[{'path':str(Path(p).relative_to(ROOT)),'domain':'easy'} for p,y in train.samples if y==saved['class_to_idx']['other']]
 hard_manifest=json.loads((ROOT/'data/hard_ood_manifest.json').read_text())
 hard=[{**r,'domain':'hard'} for r in hard_manifest['samples'] if r['split']=='train']
 # Existing feedback remains training-only and is checked against holdouts by the existing loader.
 from dataset import datasets
 with_feedback=datasets(training=True)['train'];known={r['path'] for r in easy}
 for path,label in with_feedback.samples:
  rel=str(Path(path).relative_to(ROOT))
  if label==saved['class_to_idx']['other'] and rel not in known:hard.append({'path':rel,'domain':'hard','source':'user-feedback'})
 oe_ds=Cache(easy+hard,train.augment)
 weights=[1/len(easy)]*len(easy)+[1/len(hard)]*len(hard)
 sampler=WeightedRandomSampler(weights,num_samples=len(ids),replacement=True,generator=torch.Generator().manual_seed(42))
 id_loader=DataLoader(id_ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))
 oe_loader=DataLoader(oe_ds,batch_size=32,sampler=sampler)
 valhard=Cache([r for r in hard_manifest['samples'] if r['split']=='val'])
 vlabels=np.asarray(val.targets);vid=vlabels!=saved['class_to_idx']['other']
 train_eval=CharacterDataset(splits['train']);id_eval=torch.utils.data.Subset(train_eval,ids)
 model.eval();initial_logits,_=outputs(model,id_eval)
 initial_energy=-np.logaddexp(initial_logits[:,0],initial_logits[:,1])
 margin_in=float(np.quantile(initial_energy,.9));margin_out=margin_in+2.
 initial_val_logits,_=outputs(model,val)
 baseline_id_accuracy=float((initial_val_logits[vid].argmax(1)==vlabels[vid]).mean())
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
 criterion=nn.CrossEntropyLoss();history=[];best_key=None;stale=0
 config={**vars(args),'seed':42,'initial_checkpoint_sha256':sha256(ROOT/'checkpoints/v2/best_model.pth'),
         'baseline_val_id_accuracy':baseline_id_accuracy,'maximum_val_id_accuracy_drop':.03,'margin_in':margin_in,'margin_out':margin_out,'margin_source':'90th percentile initial ID train energy; fixed 2-logit gap, not rejection thresholds',
         'oe_easy_count':len(easy),'oe_hard_count':len(hard),'oe_sampling':'equal probability mass easy/hard; sample IDs train only',
         'class_to_idx':id_mapping,'preprocessing':saved['preprocessing'],'supcon':False,'imagenet_pretrained':False}
 save_json(ROOT/'results/v3/oe_training_config.json',config);print(config,flush=True)
 for epoch in range(1,args.epochs+1):
  start=time.perf_counter();model.train();total=0.;correct=0;count=0
  for (x,y),ox in zip(id_loader,oe_loader):
   optimizer.zero_grad(set_to_none=True)
   # Mixed batch keeps BN's update domain-balanced rather than alternating incompatible statistics.
   logits,_=model.forward_outputs(torch.cat([x,ox]));il=logits[:len(x)];ol=logits[len(x):]
   ei=-torch.logsumexp(il,1);eo=-torch.logsumexp(ol,1)
   oe_loss=torch.relu(ei-margin_in).square().mean()+torch.relu(margin_out-eo).square().mean()
   loss=criterion(il,y)+args.oe_weight*oe_loss
   if not torch.isfinite(loss):raise FloatingPointError('Non-finite OE loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
   total+=loss.item()*len(y);correct+=(il.argmax(1)==y).sum().item();count+=len(y)
  model.eval();vl,ve=outputs(model,val);hl,he=outputs(model,valhard);tl,te=outputs(model,id_eval)
  gp=fit_gaussian(te,np.asarray([train.targets[i] for i in ids]),[0,1],False)
  all_emb=np.concatenate([ve,he]);all_logits=np.concatenate([vl,hl]);is_id=np.r_[vid,np.zeros(len(hl),dtype=bool)]
  d=distance(all_emb,gp);e=energy(all_logits,[0,1],1.)
  # Fixed validation-only checkpoint criterion using both real OOD scores, ID TAR >=95%.
  dt=float(np.quantile(d[is_id],.99));et=float(np.quantile(e[is_id],.99));accepted=(d<=dt)&(e<=et)
  tar=float(accepted[is_id].mean());easyfar=float(accepted[:len(val)][~vid].mean());hardfar=float(accepted[len(val):].mean())
  acc=float((vl[vid].argmax(1)==vlabels[vid]).mean());e2e=float(((vl[vid].argmax(1)==vlabels[vid])&accepted[:len(val)][vid]).mean())
  key=(tar>=.95 and acc>=baseline_id_accuracy-.03,-max(easyfar,hardfar),-((easyfar+hardfar)/2),e2e,acc)
  row={'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,'val_id_accuracy':acc,'val_id_accept':tar,'val_easy_far':easyfar,'val_hard_far':hardfar,'val_id_end_to_end_accuracy':e2e,'seconds':time.perf_counter()-start}
  history.append(row);print(row,flush=True)
  if best_key is None or key>best_key:
   best_key=key;stale=0
   torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'class_to_idx':config['class_to_idx'],'preprocessing':saved['preprocessing'],'config':config},ROOT/'checkpoints/best_model_ood.pth')
  else:stale+=1
  save_json(ROOT/'results/v3/oe_training_history.json',history)
  if stale>=7:print('Early stopping',flush=True);break
 print('OE fit complete; checkpoint chosen without reading test.',flush=True)


if __name__=='__main__':main()
