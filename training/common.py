import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from dataset import ROOT
from model import SmallCNN


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(checkpoint=ROOT / 'checkpoints/best_model_3class.pth'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = SmallCNN(num_classes=saved['model_state_dict']['head.3.weight'].shape[0])
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return model, saved
