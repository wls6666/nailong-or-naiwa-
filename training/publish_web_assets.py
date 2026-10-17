"""Publish exact ONNX/OOD bytes under SHA-256 names; never edit learned parameters.
Run standalone on existing exported files or called by export_ood.py after export.
"""
import json
from pathlib import Path
from common import ROOT,save_json,sha256


def publish():
 web=ROOT/'web';config=json.loads((web/'model_config.json').read_text())
 for field,canonical,digest_key in [('model','model.onnx','model_sha256'),('ood_config','ood_config.json','ood_config_sha256')]:
  source=web/canonical;digest=sha256(source)
  if digest!=config[digest_key]:raise RuntimeError(f'{canonical}: exported bytes do not match config; refusing to publish')
  path=Path(canonical);name=f'{path.stem}.{digest}{path.suffix}'
  (web/name).write_bytes(source.read_bytes())
  config[field]=name;config[field+'_bytes']=source.stat().st_size
 save_json(web/'model_config.json',config)
 export_path=ROOT/'results/v3/onnx_export.json'
 if export_path.exists():
  report=json.loads(export_path.read_text());report.update(config);save_json(export_path,report)
 print('Published immutable paths:',config['model'],config['ood_config'])
 return config


if __name__=='__main__':publish()
