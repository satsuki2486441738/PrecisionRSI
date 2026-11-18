"""Canonical hardware names in catalog metadata; native evidence stays unchanged."""
from copy import deepcopy
import re


NAME_FIELDS = frozenset({'hardware', 'gpu_hardware', 'title', 'label'})
SOURCE_FIELDS = frozenset({'source', 'sources', 'archive', 'evidence', 'provenance'})
H200_ALIAS = re.compile(r'\bH20Z\b', re.IGNORECASE)


def normalize_hardware_names(value):
    """Return catalog metadata with H20Z names changed to H200.

    Preserve IDs, source locators, numbers, numerical settings and native device-name
    observations. Card counts and topology are retained, not erased by a rename.
    Apply to derived catalogs only, never to frozen source payloads or receipts.
    """
    if isinstance(value, list):
        return [normalize_hardware_names(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in SOURCE_FIELDS:
                result[key] = deepcopy(item)
            elif key in NAME_FIELDS and isinstance(item, str):
                result[key] = H200_ALIAS.sub('H200', item)
            else:
                result[key] = normalize_hardware_names(item)
        return result
    return value
