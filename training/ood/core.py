"""Real shared-covariance Mahalanobis, target-logit energy, and OOD metrics."""
import json
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve


def normalize(x,enabled):
 x=np.asarray(x,dtype=np.float64)
 if enabled:x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 return x


def fit_gaussian(embedding,labels,id_indices,normalized):
 mask=np.isin(labels,id_indices);x=normalize(embedding[mask],normalized);y=np.asarray(labels)[mask]
 means=np.asarray([x[y==i].mean(0) for i in id_indices])
 residual=np.concatenate([x[y==i]-means[k] for k,i in enumerate(id_indices)])
 estimator=LedoitWolf(assume_centered=True).fit(residual)
 minimum=float(np.linalg.eigvalsh(estimator.covariance_).min())
 if minimum<=0:raise RuntimeError('Regularized covariance is not positive definite')
 return {'normalize_embedding':normalized,'embedding_dim':x.shape[1],'class_indices':list(id_indices),
         'class_centers':means.tolist(),'inverse_covariance':estimator.precision_.tolist(),
         'covariance_estimator':'LedoitWolf shared within-class residual covariance',
         'shrinkage':float(estimator.shrinkage_),'minimum_covariance_eigenvalue':minimum,'fit_id_samples':len(x)}


def distance(embedding,params):
 x=normalize(embedding,params['normalize_embedding']);means=np.asarray(params['class_centers']);precision=np.asarray(params['inverse_covariance'])
 scores=[]
 for mean in means:
  delta=x-mean;scores.append(np.einsum('ni,ij,nj->n',delta,precision,delta,optimize=True))
 return np.maximum(np.min(scores,axis=0),0.)


def probabilities(logits):
 x=np.asarray(logits,dtype=np.float64);v=np.exp(x-x.max(1,keepdims=True));return v/v.sum(1,keepdims=True)


def energy(logits,indices,temperature):
 x=np.asarray(logits,dtype=np.float64)[:,indices]/temperature;m=x.max(1)
 return -temperature*(m+np.log(np.exp(x-m[:,None]).sum(1)))


def rank_ood(scores,is_id):
 scores=np.asarray(scores);ood=~np.asarray(is_id)
 if not ood.any() or not np.asarray(is_id).any():return {'auroc':None,'aupr_ood':None,'fpr_at_95_tpr_id':None}
 fpr,tpr,_=roc_curve(is_id,-scores)
 return {'auroc':float(roc_auc_score(ood,scores)),'aupr_ood':float(average_precision_score(ood,scores)),
         'fpr_at_95_tpr_id':float(np.min(fpr[tpr>=.95]))}


def evaluate_accept(accepted,rows,logits,indices,score):
 labels=np.asarray([r['label'] for r in rows]);domains=np.asarray([r['domain'] for r in rows]);is_id=domains=='id'
 prediction=np.asarray(indices)[np.asarray(logits)[:,indices].argmax(1)]
 correct=prediction==labels;accepted=np.asarray(accepted,dtype=bool)
 out={'count':len(rows),'id_count':int(is_id.sum()),'id_true_accept_rate':float(accepted[is_id].mean()),
      'id_false_reject_rate':float((~accepted[is_id]).mean()),'closed_set_id_accuracy':float(correct[is_id].mean()),
      'id_end_to_end_accuracy':float((accepted&correct)[is_id].mean()),
      'accepted_id_accuracy':float(correct[accepted&is_id].mean()) if np.any(accepted&is_id) else None,
      'ood_false_accept_rate':float(accepted[~is_id].mean()) if np.any(~is_id) else None,
      **rank_ood(score,is_id)}
 for domain in ['easy','hard']:
  mask=domains==domain
  out[f'{domain}_ood_count']=int(mask.sum());out[f'{domain}_ood_far']=float(accepted[mask].mean()) if mask.any() else None
  out[f'{domain}_ranking']=rank_ood(np.asarray(score)[mask|is_id],is_id[mask|is_id])
 return out


def combined_score(d,e,dt,et,logic,ds,es,p=None,pt=None):
 # Zero is the calibrated boundary. Positive means reject. Scaling only changes ranking, not decisions.
 a=(np.asarray(d)-dt)/ds;b=(np.asarray(e)-et)/es
 result=np.maximum(a,b) if logic=='or' else np.minimum(a,b)
 if pt is not None:result=np.maximum(result,(np.asarray(p)-pt)/max(pt,1e-6))
 return result


def load_features(folder,split):
 folder=Path(folder);data=np.load(folder/f'{split}.npz');meta=json.loads((folder/f'{split}.json').read_text())
 return data,meta
