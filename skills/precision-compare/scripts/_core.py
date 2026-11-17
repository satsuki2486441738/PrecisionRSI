"""Resolve canonical execution tools in a checkout or a portable skill export."""
import importlib
import importlib.util
from pathlib import Path
import sys


def load(name):
    if 'tools' not in sys.modules:
        scripts = Path(__file__).resolve().parent
        candidates = (scripts / 'tools', scripts.parents[2] / 'tools')
        package = next((p for p in candidates if (p / '__init__.py').is_file()), None)
        if package is None:
            raise ImportError('tools is missing; export skills with '
                              'python -m scripts.install --output DIRECTORY')
        spec = importlib.util.spec_from_file_location('tools', package / '__init__.py',
                                                     submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['tools'] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules['tools']
            raise
    return importlib.import_module('tools.' + name)
