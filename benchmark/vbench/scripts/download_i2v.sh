#!/usr/bin/env bash
# Keep crop.zip intentionally: external inputs and downloads do not belong in git.
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash $0 DEST_DIR (external to this repository; requires Python with gdown)"
    exit 0
fi
if [[ $# != 1 ]]; then echo "Usage: bash $0 DEST_DIR" >&2; exit 2; fi
py=${PYTHON:-python3}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dest=$("$py" - "$1" "$repo" <<'PY'
from pathlib import Path
import sys
dest, repo = (Path(p).expanduser().resolve() for p in sys.argv[1:])
if dest.is_relative_to(repo):
    raise SystemExit('DEST_DIR must be outside the repository; downloaded media are external inputs')
print(dest)
PY
)
mkdir -p -- "$dest"
archive="$dest/crop.zip"
if [[ ! -f "$archive" ]]; then
    # Use the selected interpreter, not an unrelated gdown executable on PATH.
    "$py" -m gdown 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ -O "$archive.part"
    mv -- "$archive.part" "$archive"
fi
"$py" - "$archive" "$dest" <<'PY'
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile

archive, destination = map(Path, sys.argv[1:])
destination = destination.resolve(strict=True)
with zipfile.ZipFile(archive) as source:
    members = []
    # Validate the entire central directory before writing even the first file.
    for member in source.infolist():
        name = member.filename
        relative = PurePosixPath(name)
        if (not name or '\\' in name or '\0' in name or relative.is_absolute()
                or '..' in relative.parts or (relative.parts and ':' in relative.parts[0])):
            raise SystemExit(f'Unsafe ZIP member: {name!r}')
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise SystemExit(f'Non-regular ZIP member (including symlinks): {name!r}')
        target = destination.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(destination) or target == destination:
            raise SystemExit(f'ZIP member escapes destination: {name!r}')
        members.append((member, target))
    for member, target in members:
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
images = (destination / 'crop' / '16-9').resolve(strict=True)
if not images.is_relative_to(destination) or not images.is_dir():
    raise SystemExit('Archive did not provide a safe crop/16-9 directory')
print(f'Retained download: {archive}', file=sys.stderr)
print(images)
PY