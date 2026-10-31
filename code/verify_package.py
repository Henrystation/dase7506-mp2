"""Verify all packaged payload hashes without importing a model or dataset."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path)!=digest:
            raise AssertionError('Missing or modified packaged file: '+name)
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:raise AssertionError(name)
    print(json.dumps(dict(status='passed',files=len(manifest['files']),
                         version=manifest['version'],checkpoint_bytes=manifest['checkpoint_uncompressed_bytes'])))
if __name__=='__main__':main()
