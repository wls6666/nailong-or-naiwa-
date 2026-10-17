"""Vendor a pinned ONNX Runtime Web build; static hosting needs no npm or CDN."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.22.0'
FILES = ['dist/ort.wasm.min.js', 'dist/ort-wasm-simd-threaded.mjs',
         'dist/ort-wasm-simd-threaded.wasm']


def main():
    metadata = json.load(urllib.request.urlopen(f'https://registry.npmjs.org/onnxruntime-web/{VERSION}'))
    payload = urllib.request.urlopen(metadata['dist']['tarball']).read()
    integrity = 'sha512-' + base64.b64encode(hashlib.sha512(payload).digest()).decode()
    if integrity != metadata['dist']['integrity']:
        raise RuntimeError('npm package integrity mismatch')
    target = ROOT / 'web/assets/vendor/onnxruntime-web'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for filename in FILES:
            content = archive.extractfile('package/' + filename).read()
            destination = target / Path(filename).name
            destination.write_bytes(content)
            hashes[destination.name] = hashlib.sha256(content).hexdigest()
            print(destination.name, len(content), 'bytes')
        licenses = [n for n in archive.getnames() if Path(n).name.lower() in ['license', 'license.txt', 'thirdpartynotices.txt']]
        if not licenses:
            # npm 1.22.0 omits notices; retain the exact upstream release notices.
            for notice in ['LICENSE', 'ThirdPartyNotices.txt']:
                url = f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/{notice}'
                (target / notice).write_bytes(urllib.request.urlopen(url).read())
        for name in licenses:
            (target / Path(name).name).write_bytes(archive.extractfile(name).read())
    (target / 'version.json').write_text(json.dumps({'package': 'onnxruntime-web', 'version': VERSION,
                                                  'integrity': integrity, 'sha256': hashes}, indent=2) + '\n')


if __name__ == '__main__':
    main()
