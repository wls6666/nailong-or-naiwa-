"""Frozen-test comparisons, identity-clustered uncertainty, no retuning."""
import json,csv
from collections import defaultdict
import numpy as np
from common import ROOT,save_json
from v4.common import BASE


def read_predictions(name):return list(csv.DictReader((BASE/name/'predictions.csv').open()))


def report():
 names=['v3_baseline','mobilenet_frozen','mobilenet_partial_ft','mobilenet_frozen_128_control'];metrics=[json.loads((BASE/n/'metrics.json').read_text()) for n in names]
 baseline=metrics[0];base_rows=[r for r in read_predictions('v3_baseline') if r['domain']=='hard'];groups=sorted({r['identity_group'] for r in base_rows})
 rng=np.random.default_rng(42);bootstrap=rng.integers(0,len(groups),(5000,len(groups)));confidence={}
 for name,stats in zip(names[1:],metrics[1:]):
  other={r['path']:r for r in read_predictions(name) if r['domain']=='hard'};per=defaultdict(list)
  for r in base_rows:per[r['identity_group']].append((int(r['accepted']=='True'),int(other[r['path']]['accepted']=='True')))
  a=np.asarray([[sum(x[0] for x in per[g]),sum(x[1] for x in per[g]),len(per[g])] for g in groups]);sampled=a[bootstrap].sum(1);improvement=sampled[:,0]/sampled[:,2]-sampled[:,1]/sampled[:,2]
  lo,hi=np.quantile(improvement,[.025,.975]);delta=baseline['hard_ood_far']-stats['hard_ood_far']
  success=bool(delta>=.10 and lo>0 and baseline['id_end_to_end_accuracy']-stats['id_end_to_end_accuracy']<=.03 and stats['id_false_reject_rate']-baseline['id_false_reject_rate']<=.03 and stats['easy_ood_far']-baseline['easy_ood_far']<=.03)
  confidence[name]={'hard_far_improvement':delta,'identity_cluster_bootstrap_95ci':[float(lo),float(hi)],'replicates':5000,'seed':42,'success_rule_met':success,'note':'Paired identity-cluster bootstrap, art styles shared. Does not represent uncertainty across new source datasets or training seeds.'}
 selected=json.loads((BASE/'candidate_selection.json').read_text())['candidate']['name'];chosen=next(r for r in metrics if r['model']==selected)
 # Report a clear adoption conclusion. Test does not select the checkpoint/detector/candidate.
 if confidence[selected]['success_rule_met']:conclusion='A'
 elif chosen['hard_ood_far']>baseline['hard_ood_far']+.03:conclusion='C'
 else:conclusion='B'
 out={'models':metrics,'full_ft':{'status':'N/A','reason':'Optional full FT not run; core frozen/partial comparison is sufficient for this phase.'},'paired_hard_uncertainty':confidence,'candidate_selected_using_validation':selected,'conclusion':conclusion,'production_replaced':False,'recommendation':'V4 Phase 2 Hard Negative Mining; no SupCon in this phase','causal_limitations':json.loads((ROOT/'experiments/v4_phase1/protocol.json').read_text())['limitation']}
 save_json(BASE/'comparison.json',out);save_json(BASE/'mobilenet_full_ft.json',out['full_ft'])
 fields=['model','input_size','closed_set_id_accuracy','id_end_to_end_accuracy','id_macro_f1','easy_ood_far','hard_ood_far','auroc','aupr_ood','fpr_at_95_tpr_id','id_false_reject_rate','checkpoint_bytes']
 with (BASE/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(metrics)
 lines=['# V4 Phase 1 — fixed independent cartoon holdout','', 'Hard-OOD FAR is the primary endpoint. All detectors/candidate were chosen on validation only.','', '| Model | Input | ID closed acc | ID acc incl. reject | Easy FAR | **Hard FAR** | AUROC | FPR@95TPR | ID FRR |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 for r in metrics:lines.append('| '+r['model']+' | '+str(r['input_size'])+' | '+' | '.join(f'{r[k]*100:.2f}%' for k in ['closed_set_id_accuracy','id_end_to_end_accuracy','easy_ood_far','hard_ood_far','auroc','fpr_at_95_tpr_id','id_false_reject_rate'])+' |')
 lines+=['| MobileNet FullFT | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |','','Validation-selected candidate: `'+selected+'`. Conclusion **'+conclusion+'**. Production unchanged.','', 'ID closed accuracy excludes rejection; ID acc incl. reject counts rejected target images as errors. AUROC/AUPR positive OOD. FPR@95TPR: OOD acceptance at at least95% ID acceptance on test ROC, diagnostic only, not threshold tuning.','', 'The benchmark is cartoon/3D emoji art, not all commercial characters/plush photographs. Same rendered source styles remain. V3 received historical OE; new MobileNets use ID CE. Resolution control does not remove training-objective/history confounding.']
 (BASE/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':report()
