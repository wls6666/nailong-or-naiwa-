"""Reproducible multi-source Other data, preserving every original target image.
Source train pools -> train/val; official test pools -> test. No silent source fallback.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder, OxfordIIITPet
from torchvision.datasets.utils import download_url, extract_archive
from dataset import ROOT, DEFAULT_DATA, discover_splits, load_rgb

SOURCES = {
    'cifar100': {'url': 'https://www.cs.toronto.edu/~kriz/cifar.html',
                 'mirror': 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz',
                 'archive_sha256': '085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28',
                 'license': 'Official research dataset; cite Krizhevsky 2009. No explicit blanket image license on source page; do not redistribute source images.'},
    'oxford_pets': {'url': 'https://www.robots.ox.ac.uk/~vgg/data/pets/',
                    'license': 'CC BY-SA 4.0; copyright remains with original image owners; Parkhi et al., Cats and Dogs, CVPR 2012.'},
}


def digest(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_order(ds, seed):
    labels = ds.targets if hasattr(ds, 'targets') else ds._labels
    groups = defaultdict(list)
    for i, label in enumerate(labels): groups[int(label)].append(i)
    rng = random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    classes = sorted(groups); rng.shuffle(classes)
    result = []
    for j in range(max(map(len, groups.values()))):
        for label in classes:
            if j < len(groups[label]): result.append(groups[label][j])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/source_cache')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=int, default=600)
    parser.add_argument('--val', type=int, default=150)
    parser.add_argument('--test', type=int, default=150)
    parser.add_argument('--rebuild', action='store_true', help='Replace only generated Other directories, never target images')
    args = parser.parse_args()
    paths = discover_splits(args.data)
    manifest_path = ROOT / 'data/other_manifest.json'
    config = {'seed': args.seed, 'counts': {s: getattr(args, s) for s in paths}, 'sources': list(SOURCES),
              'format': 'RGB JPEG quality=95 subsampling=0, native dimensions; shared tensor resize for all classes'}
    if manifest_path.exists() and not args.rebuild:
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config: raise RuntimeError('Existing manifest configuration differs; use --rebuild explicitly')
        for row in manifest['samples']:
            p = ROOT / row['path']
            if not p.exists() or file_hash(p) != row['file_sha256']: raise RuntimeError(f'Generated image missing/modified: {p}')
        for row in manifest['original_target_files']:
            if file_hash(ROOT / row['path']) != row['sha256']: raise RuntimeError('Original target image changed')
        (ROOT / 'results/other_manifest.json').write_text(manifest_path.read_text())
        print('Existing manifest verified:', manifest['statistics'], flush=True)
        return
    original, seen = [], {}
    for split, folder in paths.items():
        for category in ['nailong', 'naiwa']:
            for p in sorted((folder / category).rglob('*')):
                if not p.is_file(): continue
                image = load_rgb(p)
                h = digest(image)
                if h in seen and seen[h] != split: raise RuntimeError(f'Original decoded-pixel cross-split duplicate: {p}')
                seen[h] = split
                original.append({'path': str(p.relative_to(ROOT)), 'sha256': file_hash(p), 'pixel_sha256': h, 'split': split})
    for split, folder in paths.items():
        dest = folder / 'other'
        if dest.exists():
            if not args.rebuild: raise RuntimeError(f'{dest} exists without a validated manifest; refusing to overwrite')
            shutil.rmtree(dest)
        dest.mkdir()
    args.cache.mkdir(parents=True, exist_ok=True)
    samples, skipped = [], []
    for source_idx, name in enumerate(SOURCES):
        print('Downloading/loading', name, flush=True)
        if name == 'cifar100':
            archive = args.cache / 'cifar100.tgz'
            if not archive.exists():
                download_url(SOURCES[name]['mirror'], str(args.cache), filename='cifar100.tgz')
            if file_hash(archive) != SOURCES[name]['archive_sha256']:
                raise RuntimeError('CIFAR100 mirror archive SHA256 mismatch; remove corrupt cached archive and retry')
            if not (args.cache / 'cifar100/train').exists():
                extract_archive(str(archive), str(args.cache))
            pools = {s: ImageFolder(str(args.cache / 'cifar100' / s)) for s in ['train', 'test']}
        else:
            pools = {'train': OxfordIIITPet(str(args.cache), split='trainval', download=True),
                     'test': OxfordIIITPet(str(args.cache), split='test', download=True)}
        order = {k: balanced_order(ds, args.seed + source_idx * 101 + (k == 'test')) for k, ds in pools.items()}
        offset = {'train': 0, 'test': 0}
        for split in ['train', 'val', 'test']:
            count = getattr(args, split) // len(SOURCES) + int(source_idx < getattr(args, split) % len(SOURCES))
            pool = 'test' if split == 'test' else 'train'
            accepted = 0
            while accepted < count:
                if offset[pool] >= len(order[pool]): raise RuntimeError(f'Not enough unique samples in {name}/{pool}')
                idx = order[pool][offset[pool]]; offset[pool] += 1
                image, label = pools[pool][idx]
                image = ImageOps.exif_transpose(image).convert('RGB')
                source_id = str(Path(pools[pool].samples[idx][0]).relative_to(args.cache / 'cifar100')) if name == 'cifar100' else pools[pool]._images[idx].stem
                h = digest(image)
                if h in seen:
                    skipped.append({'source': name, 'pool': pool, 'id': source_id, 'reason': 'decoded pixel duplicate'})
                    continue
                dest = paths[split] / 'other' / f'{name}_{pool}_{idx:06d}.jpg'
                image.save(dest, format='JPEG', quality=95, subsampling=0)
                saved_h = digest(load_rgb(dest))
                if saved_h in seen:
                    dest.unlink(); skipped.append({'source': name, 'id': source_id, 'reason': 'saved pixel duplicate'}); continue
                seen[h] = seen[saved_h] = split
                source_label = pools[pool].classes[int(label)]
                samples.append({'source': name, 'source_pool': 'trainval' if name == 'oxford_pets' and pool == 'train' else pool,
                                'source_index': idx, 'source_id': source_id, 'source_label': source_label,
                                'split': split, 'path': str(dest.relative_to(ROOT)), 'native_size': list(image.size),
                                'source_pixel_sha256': h, 'pixel_sha256': saved_h, 'file_sha256': file_hash(dest)})
                accepted += 1
            print(name, split, accepted, flush=True)
    statistics = {s: dict(Counter(r['source'] for r in samples if r['split'] == s)) for s in paths}
    manifest = {'version': 1, 'config': config, 'sources': SOURCES, 'statistics': statistics,
                'original_target_files': original, 'samples': samples, 'skipped': skipped,
                'leakage_checks': {'source_identity_unique': True, 'decoded_pixel_hash_unique': True,
                                   'saved_file_hash_unique': True, 'original_target_files_unchanged': True}}
    identities = [(r['source'], r['source_pool'], r['source_id']) for r in samples]
    assert len(set(identities)) == len(samples)
    assert len(set(r['file_sha256'] for r in samples)) == len(samples)
    # Scan persisted files independently, not just source selection bookkeeping.
    pixels = {}
    for row in original + samples:
        h = digest(load_rgb(ROOT / row['path']))
        if h in pixels and pixels[h] != row['split']: raise RuntimeError('Cross-split decoded image duplicate')
        pixels[h] = row['split']
    for row in original:
        assert file_hash(ROOT / row['path']) == row['sha256']
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    # Track the reproducibility manifest, not dataset images, in the project repository.
    (ROOT / 'results/other_manifest.json').write_text(manifest_path.read_text())
    (ROOT / 'results/other_dataset_summary.json').write_text(json.dumps({k: v for k, v in manifest.items() if k not in ['samples', 'original_target_files']}, ensure_ascii=False, indent=2) + '\n')
    print('PASS: no cross-split source/byte/decoded-pixel duplicates; original targets unchanged', flush=True)
    print(json.dumps(statistics, indent=2), flush=True)
    for split, folder in paths.items():
        ds = ImageFolder(str(folder))
        print(split, 'class_to_idx=', ds.class_to_idx, 'counts=', dict(Counter(ds.classes[y] for y in ds.targets)), flush=True)


if __name__ == '__main__':
    main()
