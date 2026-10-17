"""ID-only MobileNet projection/head training, followed by controlled partial FT.
No benchmark, OOD image, test feature, or OOD calibration metric is read here.
"""
import argparse,json,sys,time
from pathlib import Path
import torch,numpy as np
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.decomposition import PCA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,seed_all,save_json,sha256
from v4.common import BASE,Images,rows,features,MobileNet,check_production


def validate(model,loader,pooled=False):
 model.eval();total=0.;correct=0;count=0
 with torch.inference_mode():
  for x,y in loader:
   logits=model.head(x)[0] if pooled else model(x);total+=nn.functional.cross_entropy(logits,y,reduction='sum').item();correct+=(logits.argmax(1)==y).sum().item();count+=len(y)
 return total/count,correct/count


def main():
 p=argparse.ArgumentParser();p.add_argument('name',choices=['mobilenet_frozen','mobilenet_partial_ft','mobilenet_frozen_128_control']);args=p.parse_args()
 torch.set_num_threads(2);seed_all(42);check_production()
 protocol=json.loads((ROOT/'experiments/v4_phase1/protocol.json').read_text());spec=next(r for r in protocol['experiments'] if r['name']==args.name)
 size=spec['size'];partial=args.name=='mobilenet_partial_ft';out=BASE/args.name;out.mkdir(parents=True,exist_ok=True);ck=ROOT/f'checkpoints/v4_phase1/{args.name}.pth'
 trainrows=rows('train');valrows=[r for r in rows('val') if r['domain']=='id'];train=Images(trainrows,size,augment=True);val=Images(valrows,size)
 if partial:
  initial=ROOT/'checkpoints/v4_phase1/mobilenet_frozen.pth';saved=torch.load(initial,map_location='cpu',weights_only=True);model=MobileNet();model.load_state_dict(saved['model_state_dict'])
 else:model=MobileNet(pretrained=True)
 for parameter in model.features.parameters():parameter.requires_grad=False
 if partial:
  for parameter in model.features[-3:].parameters():parameter.requires_grad=True
  optimizer=torch.optim.AdamW([{'params':[p for p in model.features.parameters() if p.requires_grad],'lr':spec['lr_backbone']},{'params':list(model.projection.parameters())+list(model.classifier.parameters()),'lr':spec['lr_head']}],weight_decay=1e-4)
  train_loader=DataLoader(train,batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(42));val_loader=DataLoader(val,batch_size=32)
 else:
  model.eval();plain=Images(trainrows,size);original=features(model,plain,pooled=True);vx=features(model,val,pooled=True)
  pca=PCA(n_components=128,svd_solver='full').fit(original)
  with torch.no_grad():model.projection.weight.copy_(torch.from_numpy(pca.components_));model.projection.bias.copy_(torch.from_numpy(-pca.components_@pca.mean_))
  # Four fixed mild augmented views plus original; frozen backbone features cached.
  bank=[original];labels=[np.asarray([r['label'] for r in trainrows])]
  for view in range(4):bank.append(features(model,train,pooled=True));labels.append(labels[0]);print('cached augmented view',view+1,flush=True)
  x=torch.from_numpy(np.concatenate(bank));y=torch.from_numpy(np.concatenate(labels)).long()
  train_loader=DataLoader(TensorDataset(x,y),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42));val_loader=DataLoader(TensorDataset(torch.from_numpy(vx),torch.tensor([r['label'] for r in valrows])),batch_size=64)
  optimizer=torch.optim.AdamW([{'params':model.projection.parameters(),'lr':spec['lr_projection']},{'params':model.classifier.parameters(),'lr':spec['lr_head']}],weight_decay=1e-4)
 config={**spec,'seed':42,'weights':'torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1','official_weights_sha256':sha256(Path(torch.hub.get_dir())/'checkpoints/mobilenet_v3_small-047dcff4.pth'),'freeze_strategy':'last three feature blocks unfrozen; all earlier BN eval' if partial else 'entire backbone frozen/eval; projection and classifier trainable','normalization':'ImageNet mean/std inside graph; external RGB float32 0..1','augmentation':'flip, rotation8degrees, color jitter .12/.12/.1; frozen caches 4 augmented +1 original view','train_count':len(trainrows),'val_count':len(valrows),'selection':'ID val accuracy then ID CE loss, no OOD/test use','class_to_idx':{'nailong':0,'naiwa':1},'projection_init':'frozen best checkpoint' if partial else 'PCA128, fit only ID train then trainable','initial_checkpoint_sha256':sha256(initial) if partial else None,'protocol_sha256':sha256(ROOT/'experiments/v4_phase1/protocol.json'),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters())}
 save_json(out/'training_config.json',config);print(config,flush=True)
 history=[];best=None;stale=0;start_all=time.perf_counter()
 for epoch in range(1,spec['max_epochs']+1):
  start=time.perf_counter();model.train();model.features.eval()
  if partial:model.features[-3:].train()
  total=0.;correct=0;count=0
  for x,y in train_loader:
   optimizer.zero_grad(set_to_none=True);logits=model(x) if partial else model.head(x)[0];loss=nn.functional.cross_entropy(logits,y)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);optimizer.step()
   total+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();count+=len(y)
  vl,va=validate(model,val_loader,not partial);row={'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,'val_loss':vl,'val_accuracy':va,'seconds':time.perf_counter()-start};history.append(row);print(row,flush=True)
  key=(va,-vl)
  if best is None or key>best:
   best=key;stale=0;torch.save({'model_state_dict':model.state_dict(),'input_size':size,'epoch':epoch,'class_to_idx':config['class_to_idx'],'config':config},ck)
  else:stale+=1
  save_json(out/'training_history.json',history)
  if stale>=spec['patience']:print('Early stopping',flush=True);break
 saved=torch.load(ck,map_location='cpu',weights_only=True)
 save_json(out/'training_summary.json',{'best_epoch':saved['epoch'],'best_val_accuracy':best[0],'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':sha256(ck),'checkpoint_bytes':ck.stat().st_size,'seconds':time.perf_counter()-start_all,'epochs_run':len(history),'test_read':False})
 print('TRAIN DONE',args.name,'best',saved['epoch'],best,'seconds',time.perf_counter()-start_all,flush=True);check_production()

if __name__=='__main__':main()
