"""Extract deterministic real logits/embeddings without changing the trained encoder."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import load_model,sha256,save_json
from dataset import ROOT,DEFAULT_DATA,discover_splits,CharacterDataset,load_rgb,resize_rgb


class PathsDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return resize_rgb(load_rgb(ROOT/self.rows[i]['path']))


def extract(model,ds):
    embeddings,logits=[],[]
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=32,shuffle=False):
            x=batch[0] if isinstance(batch,(tuple,list)) else batch
            l,e=model.forward_outputs(x)
            embeddings.extend(e.numpy());logits.extend(l.numpy())
    return np.asarray(logits,dtype=np.float32),np.asarray(embeddings,dtype=np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['train','val']);p.add_argument('--checkpoint',default=ROOT/'checkpoints/v2/best_model.pth');p.add_argument('--output',type=Path,default=ROOT/'results/v3/features');args=p.parse_args()
    torch.set_num_threads(2);model,saved=load_model(args.checkpoint);args.output.mkdir(parents=True,exist_ok=True)
    paths=discover_splits(DEFAULT_DATA)
    hard_manifest=ROOT/'data/hard_ood_manifest.json'
    hard=json.loads(hard_manifest.read_text()) if hard_manifest.exists() else {'samples':[]}
    for split in args.splits:
        ds=CharacterDataset(paths[split]);logits,embedding=extract(model,ds)
        rows=[{'path':str(Path(path).relative_to(ROOT)),'label':int(label),'domain':'easy' if ds.classes[label]=='other' else 'id','split':split,'class':ds.classes[label]} for path,label in ds.samples]
        extra=[r for r in hard['samples'] if r['split']==split]
        if extra:
            l,e=extract(model,PathsDataset(extra));logits=np.concatenate([logits,l]);embedding=np.concatenate([embedding,e])
            rows.extend([{**r,'label':ds.class_to_idx['other'],'domain':'hard','class':'other'} for r in extra])
        np.savez_compressed(args.output/f'{split}.npz',logits=logits,embedding=embedding,labels=np.asarray([r['label'] for r in rows]))
        save_json(args.output/f'{split}.json',{'rows':rows,'checkpoint_sha256':sha256(args.checkpoint),'class_to_idx':saved['class_to_idx'],'embedding_dim':embedding.shape[1]})
        print(split,len(rows),'logits',logits.shape,'embedding',embedding.shape,flush=True)


if __name__=='__main__':main()
