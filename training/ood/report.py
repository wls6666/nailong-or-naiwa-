"""Collect frozen evaluations without changing choices/thresholds; preserve V1/V2."""
import csv,json,sys,shutil
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save_json

def main():
 base=ROOT/'results/v3';selection=json.loads((base/'production_selection.json').read_text());records=[];operating=[]
 for experiment in selection['experiments']:
  folder=ROOT/experiment['directory'];test=json.loads((folder/'test_metrics.json').read_text());cal=json.loads((folder/'calibration.json').read_text())
  for r in test['methods']:records.append({'experiment':experiment['family'],**r})
  for target in [.92,.95,.97,.98]:
   rows=[r for r in cal['operating_points'] if r['target_id_accept']==target]
   for method in sorted({r['method'] for r in rows}):
    best=min((r for r in rows if r['method']==method),key=lambda r:(max(r['easy_ood_far'],r['hard_ood_far']),r['ood_false_accept_rate'],-r['id_end_to_end_accuracy'],-r['id_true_accept_rate']))
    operating.append({'experiment':experiment['family'],**best})
 save_json(base/'method_comparison.json',{'selection_split':'val','test_used_to_select':False,'rows':records})
 columns=['experiment','method','closed_set_id_accuracy','id_true_accept_rate','id_false_reject_rate','id_end_to_end_accuracy','accepted_id_accuracy','easy_ood_far','hard_ood_far','ood_false_accept_rate','auroc','aupr_ood','fpr_at_95_tpr_id']
 for name,data,fields in [('method_comparison',records,columns),('validation_operating_points',operating,['target_id_accept']+columns)]:
  with (base/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(data)
 chosen=selection['selected'];folder=ROOT/chosen['directory']
 for name in ['ood_scores.csv','mahalanobis_distribution.png','energy_distribution.png','embedding_pca.png']:
  shutil.copy2(folder/name,base/name)
 report=json.loads((folder/'test_metrics.json').read_text());save_json(base/'metrics.json',{'experiment':chosen['family'],**report})
 scores=list(csv.DictReader((base/'ood_scores.csv').open()));hard=[r for r in scores if r['split']=='test' and r['true_type']=='hard'];wrong=[r for r in hard if r['accepted']=='True']
 manifest=json.loads((base/'hard_ood_manifest.json').read_text());lookup={r['path']:r for r in manifest['samples']}
 analysis={'hard_test_count':len(hard),'false_accept_count':len(wrong),'errors':[{**r,'source':lookup[r['path']]['source'],'category':lookup[r['path']]['category'],'identity':lookup[r['path']]['identity_group']} for r in wrong],
  'by_source':{},'by_category':{},'warning':'Only held-out cartoon emoji identities/renderings, not a universal hard OOD benchmark.'}
 for attr,label in [('source','by_source'),('category','by_category')]:
  for group in sorted({lookup[r['path']][attr] for r in hard}):
   rows=[r for r in hard if lookup[r['path']][attr]==group];n=sum(r['accepted']=='True' for r in rows)
   analysis[label][group]={'count':len(rows),'false_accept_count':n,'far':n/len(rows)}
 save_json(base/'hard_ood_analysis.json',analysis)
 lines=['| System | Method | ID closed accuracy | ID accept | ID end-to-end accuracy | Easy FAR | Hard FAR | AUROC | FPR@95TPR |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 for r in records:
  keys=['closed_set_id_accuracy','id_true_accept_rate','id_end_to_end_accuracy','easy_ood_far','hard_ood_far','auroc','fpr_at_95_tpr_id'];lines.append('| '+r['experiment']+' | '+r['method']+' | '+' | '.join(f'{r[k]*100:.2f}%' for k in keys)+' |')
 (base/'comparison.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines));print('Hard errors:',[(r['source'],r['identity'],r['category']) for r in analysis['errors']])

if __name__=='__main__':main()
