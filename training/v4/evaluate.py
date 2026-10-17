"""Evaluation only: parameters/checkpoint selection are already immutable."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,confusion_matrix
from common import ROOT,save_json,sha256
from ood.core import evaluate_accept,probabilities,load_features,fit_gaussian,normalize
from ood.evaluate import best_rows,score_method
from v4.common import BASE,load
from v4.detector import scores


def evaluate(name):
 folder=BASE/name;train,tm=load_features(folder/'features','train');data,meta=load_features(folder/'features','test');cfg=json.loads((folder/'ood_config.json').read_text())
 assert tm['checkpoint_sha256']==meta['checkpoint_sha256']==cfg['checkpoint_sha256']
 labels=np.asarray([r['label'] for r in meta['rows']]);domains=np.asarray([r['domain'] for r in meta['rows']]);isid=domains=='id';pred=data['logits'].argmax(1)
 d,e,s=scores(data['logits'],data['embedding'],cfg);accepted=s<=0;final=np.where(accepted,pred,2)
 stats=evaluate_accept(accepted,meta['rows'],data['logits'],[0,1],s)
 stats.update({'model':name,'checkpoint_sha256':meta['checkpoint_sha256'],'test_set_sha256':meta['set_sha256'],'benchmark_sha256':sha256(ROOT/'data/hard_ood_test_manifest.json'),'input_size':meta['input_size'],'method':cfg['method'],'reject_logic':cfg['reject_logic'],
  'id_macro_f1':float(f1_score(labels[isid],final[isid],labels=[0,1],average='macro',zero_division=0)),
  'closed_set_id_macro_f1':float(f1_score(labels[isid],pred[isid],labels=[0,1],average='macro',zero_division=0)),
  'confusion_matrix':confusion_matrix(labels,final,labels=[0,1,2]).tolist(),'confusion_order':['nailong','naiwa','ood'],
  'test_used_to_tune':False})
 for i,label in enumerate(['nailong','naiwa']):
  mask=isid&(labels==i);stats[label]={'count':int(mask.sum()),'closed_accuracy':float((pred[mask]==i).mean()),'recall_including_rejection':float((final[mask]==i).mean()),'false_reject_rate':float((~accepted[mask]).mean())}
 cal=json.loads((folder/'ood/calibration.json').read_text());ablations=[]
 for row in best_rows(cal):
  if row['method'] not in ['mahalanobis','energy','distance_energy']:continue
  ac,sc=score_method(data,cal,row,train);ablations.append({'method':row['method'],'parameters_from_validation':row['parameters'],**evaluate_accept(ac,meta['rows'],data['logits'],[0,1],sc)})
 stats['ablations']=ablations
 model,size,path=load(name);stats['checkpoint_bytes']=path.stat().st_size
 save_json(folder/'metrics.json',stats);save_json(BASE/(name+'.json'),stats)
 probability=probabilities(data['logits']);outrows=[]
 for i,r in enumerate(meta['rows']):outrows.append({**{k:r.get(k,'') for k in ['path','domain','class','identity_group','source','category']},'true_label':int(labels[i]),'predicted_id':int(pred[i]),'accepted':bool(accepted[i]),'final_label':int(final[i]),'id_confidence':float(probability[i].max()),'mahalanobis':float(d[i]),'energy':float(e[i]),'ood_score':float(s[i])})
 with (folder/'predictions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=outrows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(outrows)
 # PCA only fits the ID TRAIN feature distribution, never the test set.
 pca=PCA(n_components=2).fit(normalize(train['embedding'],cfg['normalize_embedding']));xy=pca.transform(normalize(data['embedding'],cfg['normalize_embedding']));fig,ax=plt.subplots(figsize=(6.5,4.6))
 groups=np.where(isid,np.where(labels==0,'Nailong','Naiwa'),np.where(domains=='easy','Easy OOD','Hard OOD'))
 for group in ['Nailong','Naiwa','Easy OOD','Hard OOD']:
  m=groups==group;ax.scatter(xy[m,0],xy[m,1],s=12,alpha=.6,label=group)
 ax.legend();ax.set(title=name+' — PCA fit ID train only',xlabel='PC1',ylabel='PC2');fig.tight_layout();fig.savefig(BASE/f'embeddings_{name}.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,3.6))
 for ax,values,title in zip(axes,[np.log10(np.maximum(d,1e-8)),e],['log10 squared Mahalanobis','Energy']):
  for dom in ['id','easy','hard']:ax.hist(values[domains==dom],bins=22,alpha=.5,density=True,label=dom)
  ax.set_title(title);ax.legend()
 fig.tight_layout();fig.savefig(folder/'score_distribution.png',dpi=150);plt.close(fig)
 print('TEST',name,json.dumps({k:stats[k] for k in ['closed_set_id_accuracy','id_end_to_end_accuracy','easy_ood_far','hard_ood_far','auroc','fpr_at_95_tpr_id','id_false_reject_rate']}),flush=True)
 return stats
