#!/usr/bin/env python3
"""Create per-target archive containing only portable release files."""
import argparse
from pathlib import Path
import shutil
import tarfile
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--target', required=True)
p.add_argument('--platform', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--exe', required=True)
a = p.parse_args()
root = Path.cwd()
exe = root / 'target' / a.target / 'release' / a.exe
if not exe.is_file():
    raise SystemExit(f'Executable not found: {exe}')
folder = root / 'dist' / 'grassfinder'
folder.mkdir(parents=True, exist_ok=True)
for item in ('offsets.txt', 'README.md', 'LICENSE.txt'):
    shutil.copyfile(root/item, folder/item)
shutil.copy2(exe, folder/a.exe)
base = root / 'dist' / f'grassfinder-{a.tag}-{a.platform}'
if a.platform.startswith('windows'):
    target = Path(str(base) + '.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in folder.iterdir():
            z.write(f, arcname=f'grassfinder/{f.name}')
    print(target)
else:
    target = Path(str(base) + '.tar.gz')
    with tarfile.open(target, 'w:gz') as tar:
        tar.add(folder, arcname='grassfinder')
    print(target)
