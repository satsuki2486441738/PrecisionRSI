"""Resolve canonical execution tools in a checkout or a portable skill export."""
import importlib
import importlib.util
from pathlib import Path
import sys


def load(name):
    if 'precision_core' not in sys.modules:
        scripts = Path(__file__).resolve().parent
        candidates = (scripts / 'precision_core', scripts.parents[2] / 'precision_core')
        package = next((p for p in candidates if (p / '__init__.py').is_file()), None)
        if package is None:
            raise ImportError('precision_core is missing; export skills with '
                              'python -m precision_maintenance.install --output DIRECTORY')
        spec = importlib.util.spec_from_file_location('precision_core', package / '__init__.py',
                                                     submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['precision_core'] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules['precision_core']
            raise
    return importlib.import_module('precision_core.' + name)
