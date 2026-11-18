"""Export standalone skills, bundling the canonical execution package once.

python -m scripts.install --output /path/to/client/skills
Existing destinations are rejected. No model runtime or device is used.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

from tools.community_search import load_index

ROOT = Path(__file__).resolve().parents[1]


def install_skills(source, destination):
    """Copy selected skill folders and include execution tools in Compare."""
    source, destination = Path(source), Path(destination)
    folders = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    snapshot = ROOT / 'community/site/data/search-index.json'
    if any(folder.name == 'precision-compare' for folder in folders):
        metadata, _ = load_index(snapshot)
        for source_pin in metadata['sources']:
            source_path = ROOT / 'community/site' / source_pin['file']
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_pin['sha256']:
                raise ValueError('Stale search snapshot; run python -B -m scripts export-search --refresh')
    for folder in folders:
        if (destination / folder.name).exists():
            raise FileExistsError(destination / folder.name)
    for folder in folders:
        shutil.copytree(folder, destination / folder.name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if folder.name == 'precision-compare':
            shutil.copytree(ROOT / 'tools',
                            destination / folder.name / 'scripts/tools',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.md'))
            target = destination / folder.name / 'scripts/data/community-search.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install_skills(ROOT / 'skills', args.output)
    print('Exported standalone skills to ' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
