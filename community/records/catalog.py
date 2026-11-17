"""Index immutable public records and validate explicit, evidence-bound relations.

Catalog(site) reads published JSON only. find_comparisons matches hardware and
library on the same side; related returns declared links without inferring cause.
An empty relation set is valid and makes no claim that historical cases are linked.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

# Direction and endpoint kinds are part of the relation contract.
RELATIONS = {
    'investigates': ('investigation', 'comparison'),
    'confirms': ('investigation', 'issue'),
    'addresses': ('candidate', 'issue'),
    'verifies': ('verification', 'candidate'),
    'uses_workload': ('verification', 'workload'),
    'documents': ('issue', 'repair'),
    'measures': ('verification', 'comparison'),
    'derived_from': ('measurement', 'comparison'),
}
ENTITY_KINDS = {'investigation', 'issue', 'candidate', 'verification', 'workload'}
SLUG = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*\Z')
HASH = re.compile(r'[a-f0-9]{64}\Z')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('non-finite JSON number: ' + value)
    def finite(value):
        result = float(value)
        _require(math.isfinite(result), 'non-finite JSON number: ' + value)
        return result
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                          parse_constant=invalid, parse_float=finite)
    return document, hashlib.sha256(raw).hexdigest()


class Catalog:
    """Read-only registry of comparisons, repairs, measurement views and links."""

    def __init__(self, site):
        self.site = Path(site).resolve()
        self._records = {}
        self._links = []
        for file, field, kind in (
                ('evaluations.json', 'evaluations', 'comparison'),
                ('repairs.json', 'repairs', 'repair'),
                ('differences.json', 'records', 'measurement')):
            relative = 'data/' + file
            document, source_hash = _read(self.site / relative)
            _require(type(document.get('schema_version')) is int and document['schema_version'] == 1,
                     'unsupported catalog schema: ' + file)
            values = document.get(field)
            _require(isinstance(values, list), 'missing collection: ' + field)
            for index, record in enumerate(values):
                _require(isinstance(record, dict), 'record must be an object')
                self._add(kind, record.get('id'), record,
                          dict(file=relative, pointer=f'/{field}/{index}', sha256=source_hash))
        path = self.site / 'data/relations.json'
        if path.exists():
            document, source_hash = _read(path)
            self._relations(document, source_hash)

    def _add(self, kind, identity, record, source):
        _require(isinstance(identity, str) and bool(identity), 'missing stable record id')
        key = (kind, identity)
        _require(key not in self._records, 'duplicate typed record id: ' + repr(key))
        self._records[key] = dict(kind=kind, id=identity, source=source, record=deepcopy(record))

    def _evidence(self, items):
        _require(isinstance(items, list) and bool(items), 'relation evidence is required')
        for item in items:
            _require(isinstance(item, dict) and set(item) == {'file', 'sha256'},
                     'evidence requires exactly file and sha256')
            file, digest = item['file'], item['sha256']
            _require(isinstance(file, str) and bool(file), 'invalid evidence file')
            relative = Path(file)
            path = (self.site / relative).resolve()
            _require(not relative.is_absolute() and path.is_relative_to(self.site) and path.is_file(),
                     'evidence file is missing or outside site')
            _require(isinstance(digest, str) and HASH.fullmatch(digest), 'invalid evidence hash')
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'evidence hash mismatch')

    def _endpoint(self, value):
        _require(isinstance(value, dict) and set(value) == {'kind', 'id'}, 'invalid relation endpoint')
        _require(isinstance(value['kind'], str) and isinstance(value['id'], str), 'invalid endpoint identity')
        key = (value['kind'], value['id'])
        _require(key in self._records, 'dangling relation endpoint: ' + repr(key))
        return key

    def _relations(self, document, source_hash):
        _require(isinstance(document, dict) and set(document) == {'schema_version', 'entities', 'links'},
                 'invalid relation document fields')
        _require(type(document['schema_version']) is int and document['schema_version'] == 1,
                 'unsupported relation schema')
        _require(isinstance(document['entities'], list) and isinstance(document['links'], list),
                 'relation entities and links must be lists')
        entity_ids = set()
        for index, entity in enumerate(document['entities']):
            _require(isinstance(entity, dict) and set(entity) == {'id', 'kind', 'scope', 'evidence'},
                     'invalid entity fields')
            _require(isinstance(entity['kind'], str) and entity['kind'] in ENTITY_KINDS, 'invalid entity kind')
            _require(isinstance(entity['id'], str) and SLUG.fullmatch(entity['id']), 'invalid entity id')
            _require(isinstance(entity['scope'], str) and bool(entity['scope'].strip()), 'entity scope required')
            _require(entity['id'] not in entity_ids, 'duplicate entity id')
            entity_ids.add(entity['id'])
            self._evidence(entity['evidence'])
            self._add(entity['kind'], entity['id'], entity,
                      dict(file='data/relations.json', pointer=f'/entities/{index}', sha256=source_hash))
        seen = set()
        edges = set()
        for link in document['links']:
            _require(isinstance(link, dict) and set(link) == {'id', 'relation', 'from', 'to', 'scope', 'evidence'},
                     'invalid link fields')
            identity, relation = link['id'], link['relation']
            _require(isinstance(identity, str) and SLUG.fullmatch(identity) and identity not in seen,
                     'invalid or duplicate link id')
            seen.add(identity)
            _require(isinstance(relation, str) and relation in RELATIONS, 'unknown relation type')
            left, right = self._endpoint(link['from']), self._endpoint(link['to'])
            _require((left[0], right[0]) == RELATIONS[relation], 'relation endpoint kind mismatch')
            _require(isinstance(link['scope'], str) and bool(link['scope'].strip()), 'link scope required')
            self._evidence(link['evidence'])
            edge = (relation, left, right)
            _require(edge not in edges, 'duplicate relation edge')
            edges.add(edge)
            self._links.append(deepcopy(link))
        # A verification record must identify the candidate and original workload.
        for kind, identity in self._records:
            if kind == 'verification':
                outgoing = {r for r, left, right in edges if left == (kind, identity)}
                _require({'verifies', 'uses_workload'} <= outgoing,
                         'verification requires a candidate and original workload')

    def get(self, kind, identity):
        """Return a detached record and its exact public source locator."""
        return deepcopy(self._records[(kind, identity)])

    def index(self):
        """Return stable typed identities and locators, without rewriting records."""
        return [dict(kind=k, id=i, source=deepcopy(v['source']))
                for (k, i), v in self._records.items()]

    def related(self, kind, identity):
        """Return only explicitly recorded incident links; no heuristic joins."""
        _require((kind, identity) in self._records, 'unknown record')
        endpoint = dict(kind=kind, id=identity)
        return deepcopy([link for link in self._links if endpoint in (link['from'], link['to'])])

    def find_comparisons(self, *, hardware=None, library=None, model=None, task=None):
        """Exact hardware/library matching on one side; library may include version."""
        result = []
        for (kind, identity), entry in self._records.items():
            if kind != 'comparison':
                continue
            record = entry['record']
            if model is not None and record.get('model', {}).get('name') != model:
                continue
            if task is not None and record.get('task') != task:
                continue
            def matches(side):
                return ((hardware is None or side.get('hardware') == hardware) and
                        (library is None or any(all(value.get(k) == v for k, v in library.items())
                                                for value in side.get('libraries', []))))
            if hardware is None and library is None or any(matches(side) for side in record.get('sides', [])):
                result.append(self.get(kind, identity))
        return result
