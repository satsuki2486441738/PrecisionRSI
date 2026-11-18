"""Validate concise public descriptions independently of experiment evidence.

New records can provide ``content`` with localized ``title`` and optional
``summary``. ``data/content.json`` supplies the same fields for historical
records. Both are keyed by a typed record identity, not by frontend conditions.
"""
from copy import deepcopy


def validate_content(value):
    if not isinstance(value, dict) or not {'title'} <= value.keys() <= {'title', 'summary', 'workflows'}:
        raise ValueError('content requires title and optional summary/workflows')
    for field, maximum in (('title', 120), ('summary', 400)):
        if field not in value:
            continue
        localized = value[field]
        if (not isinstance(localized, dict) or not localized or
                not localized.keys() <= {'zh', 'en'} or
                any(not isinstance(text, str) or not text.strip() or len(text) > maximum
                    for text in localized.values())):
            raise ValueError(f'content {field} requires zh/en text up to {maximum} characters')
    if 'workflows' in value:
        workflows = value['workflows']
        if (not isinstance(workflows, list) or any(task not in ('inference', 'sft', 'rl') for task in workflows)
                or len(workflows) != len(set(workflows))):
            raise ValueError('content workflows requires distinct inference/sft/rl values')
    return deepcopy(value)


def load_content(records, document=None):
    """Return reviewed descriptions; reject duplicate identities and stale targets."""
    result = {}
    for identity, entry in records.items():
        row = entry['record']
        value = row.get('content', row.get('details', {}).get('content'))
        if 'content' in row or 'content' in row.get('details', {}):
            result[identity] = validate_content(value)
    if document is None:
        return result
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'records'} or
            type(document['schema_version']) is not int or document['schema_version'] != 1 or
            not isinstance(document['records'], list)):
        raise ValueError('invalid community content document')
    for entry in document['records']:
        if not isinstance(entry, dict) or set(entry) != {'kind', 'id', 'content'}:
            raise ValueError('content entry requires kind, id and content')
        if not all(isinstance(entry[field], str) for field in ('kind', 'id')):
            raise ValueError('content identity must be text')
        identity = (entry['kind'], entry['id'])
        if identity not in records:
            raise ValueError('unknown content record: ' + repr(identity))
        if identity in result:
            raise ValueError('duplicate content record: ' + repr(identity))
        result[identity] = validate_content(entry['content'])
    return result


def localized(content, field, language='zh', default=''):
    values = content.get(field, {})
    return values.get(language) or values.get('en') or values.get('zh') or default
