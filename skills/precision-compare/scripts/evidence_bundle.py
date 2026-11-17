#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives in tools.evidence_bundle."""
CORE_MODULE = "evidence_bundle"
import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location('_tools_entry', Path(__file__).with_name('_core.py'))
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)
_module = _entry.load(CORE_MODULE)
if __name__ == '__main__':
    if hasattr(_module, 'main'):
        raise SystemExit(_module.main())
else:
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith('__')})
    __doc__ = _module.__doc__
    sys.modules[__name__] = _module
