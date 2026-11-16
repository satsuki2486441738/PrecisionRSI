"""Export standalone skills, bundling the canonical execution package once.

python -m precision_maintenance.install --output /path/to/client/skills
Existing destinations are rejected. No model runtime or device is used.
"""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def install_skills(source, destination):
    """Copy selected skill folders and include execution tools in Compare."""
    source, destination = Path(source), Path(destination)
    folders = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    for folder in folders:
        if (destination / folder.name).exists():
            raise FileExistsError(destination / folder.name)
    for folder in folders:
        shutil.copytree(folder, destination / folder.name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if folder.name == 'precision-compare':
            shutil.copytree(ROOT / 'precision_core',
                            destination / folder.name / 'scripts/precision_core',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install_skills(ROOT / 'skills', args.output)
    print('Exported standalone skills to ' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
