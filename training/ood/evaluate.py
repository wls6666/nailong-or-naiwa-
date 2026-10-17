"""Evaluate frozen calibration once on test; produce all ablations, scores and plots."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save_json
from ood.core import load_features,distance,energy,probabilities,combined_score,evaluate_accept,normalize


def best_rows(cal):
 result=[]
 for method in ['v2_softmax','msp','p_other','mahalanobis','energy','distance_energy','distance_energy_p_other']:
  recs=[r for r in cal['operating_points'] if r['method']==method and r['target_id_accept'] in [None,.95]]
  def key(r):
   far=[r[k] for k in ['easy_ood_far','hard_ood_far'] if r[k] is not None]
   return(max(far),r['ood_false_accept_rate'],-r['id_end_to_end_accuracy'],-r['id_true_accept_rate'])
  if recs:result.append(min(recs,key=key))
 return result


def score_method(data,cal,row,train):
 mapping=cal['class_mapping'];indices=[mapping['nailong'],mapping['naiwa']];p=row['parameters'];method=row['method']
 prob=probabilities(data['logits']);other=prob[:,mapping['other']] if data['logits'].shape[1]==3 else None
 if method=='v2_softmax':score=1-prob[:,indices].max(1);return score<=p['threshold'],score
 if method=='msp':score=1-probabilities(data['logits'][:,indices]).max(1);return score<=p['threshold'],score
 if method=='p_other':return other<=p['threshold'],other
 from ood.core import fit_gaussian
 gp=fit_gaussian(train['embedding'],train['labels'],indices,p.get('normalize_embedding',False))
 d=distance(data['embedding'],gp);e=energy(data['logits'],indices,p.get('energy_temperature',1.))
 if method=='mahalanobis':return d<=p['distance_threshold'],d
 if method=='energy':return e<=p['energy_threshold'],e
 score=combined_score(d,e,p['distance_threshold'],p['energy_threshold'],p['reject_logic'],p['distance_scale'],p['energy_scale'],other,p.get('p_other_threshold'))
 return score<=0,score


def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=ROOT/'results/v3/features');p.add_argument('--calibration',type=Path,default=ROOT/'results/v3/phase_a/calibration.json');p.add_argument('--output',type=Path,default=ROOT/'results/v3/phase_a');args=p.parse_args()
 cal=json.loads(args.calibration.read_text());train,tm=load_features(args.features,'train');test,meta=load_features(args.features,'test')
 assert cal['checkpoint_sha256']==meta['checkpoint_sha256']==tm['checkpoint_sha256']
 args.output.mkdir(parents=True,exist_ok=True);indices=[cal['class_mapping'][k] for k in ['nailong','naiwa']]
 reports=[]
 for row in best_rows(cal):
  accepted,score=score_method(test,cal,row,train)
  reports.append({'method':row['method'],'validation_parameters':row['parameters'],**evaluate_accept(accepted,meta['rows'],test['logits'],indices,score)})
 selected=cal['selected'];accepted,score=score_method(test,cal,selected,train)
 selected_report={'method':selected['method'],**evaluate_accept(accepted,meta['rows'],test['logits'],indices,score)}
 save_json(args.output/'test_metrics.json',{'selection':'Frozen validation choice; test never used for method/threshold selection','methods':reports,'selected':selected_report,'checkpoint_sha256':cal['checkpoint_sha256']})
 from ood.core import fit_gaussian
 params=selected['parameters'];gp=fit_gaussian(train['embedding'],train['labels'],indices,params.get('normalize_embedding',False))
 rows_all=[];per_split={}
 for split in ['val','test']:
  data,md=load_features(args.features,split);d=distance(data['embedding'],gp);e=energy(data['logits'],indices,params.get('energy_temperature',1.));prob=probabilities(data['logits']);a,s=score_method(data,cal,selected,train)
  pred=np.asarray(indices)[data['logits'][:,indices].argmax(1)];per_split[split]=(data,md,d,e)
  for i,r in enumerate(md['rows']):
   rows_all.append({'path':r['path'],'split':split,'true_type':r['domain'],'true_class':r['class'],'predicted_class':next(k for k,v in cal['class_mapping'].items() if v==pred[i]) if a[i] else 'other','max_softmax':float(probabilities(data['logits'][i:i+1,indices]).max()),'mahalanobis':float(d[i]),'energy':float(e[i]),'p_other':float(prob[i,2]) if prob.shape[1]==3 else '', 'ood_score':float(s[i]),'accepted':bool(a[i]),'correct':bool(pred[i]==r['label'] and a[i]) if r['domain']=='id' else bool(not a[i])})
 with (args.output/'ood_scores.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows_all[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(rows_all)
 for name,col in [('mahalanobis',2),('energy',3)]:
  fig,axes=plt.subplots(1,2,figsize=(10,3.6))
  for ax,split in zip(axes,['val','test']):
   data,md,d,e=per_split[split];values=d if name=='mahalanobis' else e
   if name=='mahalanobis':values=np.log10(np.maximum(values,1e-8))
   domains=np.asarray([r['domain'] for r in md['rows']])
   for domain in ['id','easy','hard']:ax.hist(values[domains==domain],bins=22,alpha=.45,density=True,label=domain)
   ax.set_title(split);ax.set_xlabel('log10 squared Mahalanobis' if name=='mahalanobis' else 'Energy (higher = OOD)');ax.legend()
  fig.tight_layout();fig.savefig(args.output/f'{name}_distribution.png',dpi=150);plt.close(fig)
 # PCA fit exclusively on ID train; plotting test does not refit the projection.
 train_id=np.isin(train['labels'],indices);pca=PCA(n_components=2).fit(normalize(train['embedding'][train_id],gp['normalize_embedding']))
 data,md,_,_=per_split['test'];xy=pca.transform(normalize(data['embedding'],gp['normalize_embedding']));fig,ax=plt.subplots(figsize=(6,4.5))
 groups=np.asarray([r['class'] if r['domain']=='id' else r['domain'] for r in md['rows']])
 for group in ['nailong','naiwa','easy','hard']:
  mask=groups==group;ax.scatter(xy[mask,0],xy[mask,1],s=12,alpha=.65,label=group)
 ax.legend();ax.set(xlabel='PC1 (fit ID train)',ylabel='PC2',title='Embedding PCA — illustrative, not an OOD metric');fig.tight_layout();fig.savefig(args.output/'embedding_pca.png',dpi=150);plt.close(fig)
 print(json.dumps(selected_report,indent=2))


if __name__=='__main__':main()
