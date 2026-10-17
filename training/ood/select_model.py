"""Freeze production choice from validation only, before any new test scoring."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save_json,sha256


def main():
 cases=[('custom_posthoc','results/v3/phase_a','results/v3/features','checkpoints/v2/best_model.pth'),
        ('custom_energy_oe','results/v3/phase_b','results/v3/phase_b_features','checkpoints/best_model_ood.pth'),
        ('mobilenet_frozen','results/v3/mobilenet','results/v3/mobilenet_features','checkpoints/mobilenet_ood.pth')]
 rows=[]
 for name,directory,features,checkpoint in cases:
  cal=json.loads((ROOT/directory/'calibration.json').read_text());selected=cal['selected']
  record={'family':name,'directory':directory,'features':features,'checkpoint':checkpoint,'checkpoint_sha256':sha256(ROOT/checkpoint),'validation':selected}
  record['eligible']=selected['id_true_accept_rate']>=.95 and selected['closed_set_id_accuracy']>=.9315068493150684-.03
  rows.append(record)
 def key(r):
  v=r['validation'];return(max(v['easy_ood_far'],v['hard_ood_far']),v['ood_false_accept_rate'],-v['id_end_to_end_accuracy'],-v['id_true_accept_rate'])
 custom=min((r for r in rows[:2] if r['eligible']),key=key)
 mobile=rows[2]
 # A larger pretrained control replaces the custom encoder only on a material validation improvement.
 improved=mobile['eligible'] and key(mobile)[0]<=key(custom)[0]-.05
 selected=mobile if improved else custom
 report={'selection_split':'val','test_used':False,'rule':'ID accept >=95%, closed-set ID accuracy at most 3pp below frozen baseline. Best custom method minimizes worst Easy/Hard FAR; MobileNet replaces it only if worst-domain FAR improves by at least 5pp.',
         'experiments':rows,'selected':selected}
 save_json(ROOT/'results/v3/production_selection.json',report)
 print('PRODUCTION VALIDATION CHOICE:',selected['family'],selected['validation']['method'],selected['validation']['parameters'])


if __name__=='__main__':main()
