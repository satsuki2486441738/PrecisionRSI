"""Resolve legacy maintenance entrypoints without modifying import paths."""
import importlib
import importlib.util
from pathlib import Path
import sys


def load(package, module):
    root = Path(__file__).resolve().parents[1]
    for name in ('precision_community', 'precision_maintenance'):
        if name not in sys.modules:
            folder = root / name
            spec = importlib.util.spec_from_file_location(name, folder / '__init__.py',
                                                        submodule_search_locations=[str(folder)])
            value = importlib.util.module_from_spec(spec)
            sys.modules[name] = value
            spec.loader.exec_module(value)
    return importlib.import_module(package + '.' + module)
