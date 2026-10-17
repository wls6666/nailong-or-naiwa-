"""Independent pretrained CNN control after custom-CNN OOD proves weak on validation.
Fit PCA and a real PyTorch CE head on ID train only; no test access.
"""
import json
from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from sklearn.decomposition import PCA
import numpy as np
from torch.utils.data import DataLoader
from dataset import ROOT,DEFAULT_DATA,discover_splits,CharacterDataset,PREPROCESS
from common import seed_all,save_json
from ood.mobilenet import MobileNetOOD


def features(model,ds):
 out=[]
 with torch.inference_mode():
  for x,y in DataLoader(ds,batch_size=32):out.extend(model.backbone_embedding(x).numpy())
 return np.asarray(out,dtype=np.float32)


def main():
 torch.set_num_threads(2);seed_all(42)
 pretrained=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
 model=MobileNetOOD();model.features.load_state_dict(pretrained.features.state_dict());model.eval()
 splits=discover_splits(DEFAULT_DATA);train=CharacterDataset(splits['train']);val=CharacterDataset(splits['val'])
 mapping={name:train.class_to_idx[name] for name in ['nailong','naiwa']}
 tx=features(model,train);vx=features(model,val);ty=np.asarray(train.targets);vy=np.asarray(val.targets)
 tid=np.isin(ty,list(mapping.values()));vid=np.isin(vy,list(mapping.values()))
 pca=PCA(n_components=128,svd_solver='full').fit(tx[tid])
 with torch.no_grad():
  model.projection.weight.copy_(torch.from_numpy(pca.components_));model.projection.bias.copy_(torch.from_numpy(-pca.components_@pca.mean_))
 xt=torch.from_numpy(pca.transform(tx[tid]).copy());xv=torch.from_numpy(pca.transform(vx[vid]).copy());yt=torch.from_numpy(ty[tid]).long();yv=torch.from_numpy(vy[vid]).long()
 optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=.01,weight_decay=1e-3);criterion=torch.nn.CrossEntropyLoss()
 history=[];best=None;stale=0
 config={'model_kind':'mobilenet_v3_small_pca','pretrained':'torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1','projection':'128D PCA fit only 345 ID train embeddings',
         'internal_normalization':'ImageNet mean/std inside exported graph; JS input still RGB [0,1]',
         'encoder_frozen':True,'head_objective':'PyTorch CrossEntropyLoss on ID train only','supcon':False,'oe':False,'seed':42}
 for epoch in range(1,201):
  model.classifier.train();optimizer.zero_grad(set_to_none=True);logits=model.classifier(xt);loss=criterion(logits,yt);loss.backward();optimizer.step()
  model.classifier.eval()
  with torch.inference_mode():vl=model.classifier(xv);va=(vl.argmax(1)==yv).float().mean().item();v_loss=criterion(vl,yv).item()
  row={'epoch':epoch,'train_loss':loss.item(),'val_id_accuracy':va,'val_loss':v_loss};history.append(row)
  key=(va,-v_loss)
  if best is None or key>best:
   best=key;stale=0
   torch.save({'model_kind':'mobilenet_v3_small_pca','model_state_dict':model.state_dict(),'class_to_idx':mapping,'preprocessing':PREPROCESS,'epoch':epoch,'config':config},ROOT/'checkpoints/mobilenet_ood.pth')
  else:stale+=1
  if epoch%20==0:print(row,flush=True)
  if stale>=40:break
 save_json(ROOT/'results/v3/mobilenet_training.json',{'config':config,'history':history,'best_val_id_accuracy':best[0]})
 print('MobileNet real CE head fitted; best validation ID accuracy',best[0],flush=True)


if __name__=='__main__':main()
