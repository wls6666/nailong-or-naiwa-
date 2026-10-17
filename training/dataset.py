"""Discover original splits; shared deterministic RGB/resize/CHW preprocessing."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/NaiwaOrNailong-dataset'
SIZE = 128
PREPROCESS = {
    'size': SIZE, 'channels': 'RGB', 'layout': 'NCHW', 'dtype': 'float32',
    'range': [0, 1], 'normalization': None,
    'resize': 'bilinear-half-pixel-clamp-float-no-round',
    'exif_orientation': True, 'alpha_background': 'white',
}


def discover_splits(root):
    root = Path(root).resolve()
    matches = []
    for candidate in [root, root / 'data', *sorted(root.glob('*/data'))]:
        if not candidate.is_dir():
            continue
        children = {p.name.lower(): p for p in candidate.iterdir() if p.is_dir()}
        val = children.get('val', children.get('validation', children.get('valid')))
        if 'train' in children and 'test' in children and val:
            matches.append({'train': children['train'], 'val': val, 'test': children['test']})
    unique = {str(m['train']): m for m in matches}
    if len(unique) != 1:
        raise ValueError(f'Expected one existing train/val(or validation)/test layout in {root}, found {list(unique)}')
    return next(iter(unique.values()))


def load_rgb(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, rgba).convert('RGB')
        return image.convert('RGB')


def resize_rgb(image):
    """Explicit half-pixel bilinear, mirrored in web/preprocess.js; no PIL/Canvas resize ambiguity."""
    pixels = np.asarray(image, dtype=np.float32)
    h, w = pixels.shape[:2]
    x = np.maximum(0, (np.arange(SIZE, dtype=np.float64) + .5) * w / SIZE - .5)
    y = np.maximum(0, (np.arange(SIZE, dtype=np.float64) + .5) * h / SIZE - .5)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (x - x0)[None, :, None], (y - y0)[:, None, None]
    top = pixels[y0[:, None], x0] * (1 - wx) + pixels[y0[:, None], x1] * wx
    bottom = pixels[y1[:, None], x0] * (1 - wx) + pixels[y1[:, None], x1] * wx
    out = (top * (1 - wy) + bottom * wy) / 255.0
    return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32))


class CharacterDataset(ImageFolder):
    def __init__(self, path, augment=False):
        super().__init__(str(path), loader=load_rgb)
        if set(self.classes) != {'nailong', 'naiwa'}:
            raise ValueError(f'Unexpected classes: {self.classes}')
        # Cache deterministic tensors, keeping CPU training practical without changing source images.
        self.cache = [resize_rgb(load_rgb(p)) for p, _ in self.samples]
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8, interpolation=transforms.InterpolationMode.BILINEAR, fill=1),
            transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.1),
        ]) if augment else None

    def __getitem__(self, index):
        tensor = self.cache[index]
        if self.augment:
            tensor = self.augment(tensor)
        return tensor, self.samples[index][1]


def datasets(root=DEFAULT_DATA, training=False):
    paths = discover_splits(root)
    result = {name: CharacterDataset(path, augment=training and name == 'train')
              for name, path in paths.items()}
    mapping = result['train'].class_to_idx
    if any(ds.class_to_idx != mapping for ds in result.values()):
        raise ValueError('Split class mappings differ')
    print('class_to_idx:', mapping, flush=True)
    return result


def audit(root=DEFAULT_DATA):
    paths = discover_splits(root)
    report = {'split_paths': {k: str(v.relative_to(Path(root).resolve())) for k, v in paths.items()},
              'splits': {}, 'duplicate_bytes': [], 'preprocessing': PREPROCESS,
              'source_url': 'https://github.com/CristinXie/NaiwaOrNailong-dataset'}
    if (Path(root) / '.git').exists():
        report['source_commit'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    seen = {}
    for name, path in paths.items():
        ds = ImageFolder(str(path))
        counts = Counter()
        for file, label in ds.samples:
            image = load_rgb(file)
            image.load()
            counts[ds.classes[label]] += 1
            digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            if digest in seen:
                report['duplicate_bytes'].append([seen[digest], str(Path(file).relative_to(Path(root)))])
            seen[digest] = str(Path(file).relative_to(Path(root)))
        report['splits'][name] = dict(counts)
        report['class_to_idx'] = ds.class_to_idx
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/dataset_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    audit(parser.parse_args().data)
