"""Test the Updater with a local file:// metadata and asset.

This script:
- creates packaging/test_assets/HGCameraCounter-test.zip
- writes metadata.json pointing to the file URI and containing SHA256
- runs Updater.check_for_update/download/verify/stage and prints results
"""
from pathlib import Path
import hashlib
import json
from shared.updater import Updater

root = Path(__file__).resolve().parent.parent
assets_dir = root / 'packaging' / 'test_assets'
assets_dir.mkdir(parents=True, exist_ok=True)

asset_path = assets_dir / 'HGCameraCounter-test.zip'
with open(asset_path, 'wb') as f:
    f.write(b'Test update asset content')

# compute sha256
h = hashlib.sha256()
with open(asset_path, 'rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
sha = h.hexdigest()

metadata = {
    'version': '0.0.1-test',
    'notes': 'Test release',
    'assets': [
        {
            'name': asset_path.name,
            'url': asset_path.resolve().as_uri(),
            'sha256': sha
        }
    ]
}

metadata_path = assets_dir / 'metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print('Asset created:', asset_path)
print('Metadata created:', metadata_path)

# Run updater
updater = Updater()
md_url = metadata_path.resolve().as_uri()
print('Checking metadata at', md_url)
meta = updater.check_for_update(md_url)
print('Metadata read:', meta.get('version'))
asset = updater.select_primary_asset(meta)
print('Selected asset:', asset)

downloaded = updater.download_asset(asset['url'], name=asset['name'])
print('Downloaded to:', downloaded)

ok = updater.verify_sha256(downloaded, asset['sha256'])
print('SHA256 ok?', ok)
if ok:
    staged = updater.stage_update(downloaded)
    print('Staged to:', staged)
else:
    print('Verification failed')
