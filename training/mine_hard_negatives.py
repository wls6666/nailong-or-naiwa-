"""Rank only hard TRAIN instances using the frozen deployed V2 model's real errors."""
import json
import numpy as np
from pathlib import Path
from dataset import ROOT
from common import save_json
from ood.core import load_features,probabilities


def main():
 data,meta=load_features(ROOT/'results/v3/features','train')
 assert data['logits'].shape[1]==3,'Mining requires frozen V2 features, not test or a new model'
 config=json.loads((ROOT/'checkpoints/v2/model_config.json').read_text())
 mapping=config['class_to_idx'];p=probabilities(data['logits']);target=p[:,[mapping['nailong'],mapping['naiwa']]].max(1)
 rows=[]
 for i,r in enumerate(meta['rows']):
  if r['domain']!='hard':continue
  assert r['split']=='train'
  accepted=bool(target[i]>=config['reject_threshold'])
  rows.append({**r,'v2_false_accept':accepted,'v2_max_target_softmax':float(target[i]),'v2_p_other':float(p[i,mapping['other']]),'sampling_priority':4. if accepted else 1.})
 rows.sort(key=lambda r:(-int(r['v2_false_accept']),-r['v2_max_target_softmax'],r['path']))
 save_json(ROOT/'results/v3/hard_negative_mining.json',{'split':'train','checkpoint_sha256':meta['checkpoint_sha256'],'sample_count':len(rows),'v2_false_accept_count':sum(r['v2_false_accept'] for r in rows),'rows':rows,'test_used':False,'validation_used':False})
 print('Real hard TRAIN samples',len(rows),'V2 false accepts',sum(r['v2_false_accept'] for r in rows))


if __name__=='__main__':main()
