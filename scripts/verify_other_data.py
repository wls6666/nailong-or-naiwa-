"""Independently verify generated datasets, original splits, and manifest identities."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from dataset import ROOT, load_rgb
from common import sha256


def pixel_hash(path):
    image = load_rgb(path)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'data/other_manifest.json').read_text())
    seen_files, seen_pixels, seen_ids = {}, {}, set()
    statistics = Counter()
    for row in manifest['original_target_files'] + manifest['samples']:
        path = ROOT / row['path']
        expected = row.get('file_sha256', row.get('sha256'))
        assert sha256(path) == expected, path
        for key, seen in [(sha256(path), seen_files), (pixel_hash(path), seen_pixels)]:
            assert key not in seen or seen[key] == row['split'], f'Cross-split duplicate: {path}'
            seen[key] = row['split']
        if 'source' in row:
            identity = (row['source'], row['source_pool'], row['source_id'])
            assert identity not in seen_ids, identity
            seen_ids.add(identity); statistics[(row['source'], row['split'])] += 1
    for row in manifest['samples']:
        relative = Path(row['path'])
        assert relative.parent.name == 'other'
        # The source pool split must agree with the destination split isolation rule.
        assert (row['source_pool'] == 'test') == (row['split'] == 'test')
    report = {'status': 'PASS', 'original_targets': len(manifest['original_target_files']),
              'other_images': len(manifest['samples']), 'statistics': {f'{s}/{split}': n for (s,split), n in statistics.items()},
              'manifest_sha256': sha256(ROOT / 'data/other_manifest.json'),
              'byte_and_pixel_cross_split_duplicates': 0, 'duplicate_source_ids': 0,
              'limitation': 'Exact decoded-pixel/source/byte checks do not prove absence of all perceptually similar scenes.'}
    (ROOT / 'results/other_data_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
