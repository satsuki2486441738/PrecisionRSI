"""Index immutable public records and validate explicit, evidence-bound relations.

Catalog(site) reads published JSON only. find_comparisons matches hardware and
library on the same side; related returns declared links without inferring cause.
An empty relation set is valid and makes no claim that historical cases are linked.
"""
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .models import model_facets, model_group


def _matches_model(record, model, model_size):
    if model is None and model_size is None:
        return True
    identity = record.get('model', {})
    architecture, size = model_facets(identity)
    return ((model is None or model == identity.get('name') or model.casefold() == architecture.casefold()
             or model.casefold() == model_group(identity, architecture).casefold()
             or model.casefold() == 'qwen2.5' and architecture == 'Qwen2') and
            (model_size is None or model_size.casefold() == size.casefold()))

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
    'includes': ('case', {'comparison', 'repair', 'measurement', 'investigation',
                          'issue', 'candidate', 'verification', 'workload'}),
}
ENTITY_KINDS = {'case', 'investigation', 'issue', 'candidate', 'verification', 'workload'}
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
        self._sources = {}
        for file, field, kind in (
                ('evaluations.json', 'evaluations', 'comparison'),
                ('repairs.json', 'repairs', 'repair'),
                ('differences.json', 'records', 'measurement')):
            relative = 'data/' + file
            document, source_hash = _read(self.site / relative)
            self._sources[relative] = (document, source_hash)
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
        from .content import load_content
        path = self.site / 'data/content.json'
        document, source_hash = _read(path) if path.exists() else (None, None)
        self._content = load_content(self._records, document)
        self._content_hash = source_hash

    def content(self, kind, identity):
        """Return the concise description for a typed record identity."""
        if (kind, identity) not in self._records:
            raise KeyError((kind, identity))
        return deepcopy(self._content.get((kind, identity), {}))

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

    def resolve_source(self, source):
        """Resolve a hash-bound JSON Pointer from a frozen local evidence snapshot.

        Sources use file/sha256/pointer for frozen evidence, or kind/id/pointer
        for existing comparison/repair/measurement records. Stable record IDs
        allow appendable catalogs to grow without invalidating earlier links.
        """
        _require(isinstance(source, dict), 'source must be an object')
        if set(source) == {'kind', 'id', 'pointer'}:
            _require(source['kind'] in ('comparison', 'repair', 'measurement') and
                     isinstance(source['id'], str), 'invalid legacy record source')
            key = (source['kind'], source['id'])
            _require(key in self._records, 'unknown legacy source record')
            document = self._records[key]['record']
        else:
            _require(set(source) == {'file', 'sha256', 'pointer'},
                     'source requires file/sha256/pointer or kind/id/pointer')
            file, digest = source['file'], source['sha256']
            _require(isinstance(file, str) and file.startswith('evidence/') and file.endswith('.json'),
                     'source must reference frozen JSON evidence')
            path = (self.site / file).resolve()
            _require(path.is_relative_to(self.site / 'evidence') and path.is_file(),
                     'source is missing or outside evidence')
            _require(isinstance(digest, str) and HASH.fullmatch(digest), 'invalid source hash')
            if file not in self._sources:
                self._sources[file] = _read(path)
            document, actual = self._sources[file]
            _require(actual == digest, 'source hash mismatch')
        pointer = source['pointer']
        _require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')),
                 'invalid source pointer')
        value = document
        try:
            for token in pointer.split('/')[1:] if pointer else ():
                _require(re.search(r'~(?![01])', token) is None, 'invalid pointer escape')
                key = token.replace('~1', '/').replace('~0', '~')
                if isinstance(value, list):
                    _require(re.fullmatch(r'0|[1-9][0-9]*', key) is not None,
                             'invalid source array index')
                    value = value[int(key)]
                else:
                    _require(isinstance(value, dict), 'source pointer traverses a scalar')
                    value = value[key]
        except (KeyError, IndexError) as exc:
            raise ValueError('unresolved source pointer') from exc
        return deepcopy(value)

    def _details(self, entity):
        """Check optional lifecycle metadata without inventing acceptance or fixes."""
        if 'details' not in entity:
            _require(entity['kind'] != 'case', 'case details are required')
            return
        details = entity['details']
        common = {'title', 'sources'}
        fields = {
            'case': common | {'request'},
            'issue': common | {'classification'},
            'candidate': common | {'solution_type', 'target', 'applicability',
                                    'evidence_level', 'decision'},
            'verification': common | {'method', 'outcome', 'acceptance'},
            'workload': common,
            'investigation': common,
        }[entity['kind']]
        optional = {'content', 'limitations'} | ({'category'} if entity['kind'] == 'case' else set())
        _require(isinstance(details, dict) and fields <= set(details) <= fields | optional,
                 'invalid lifecycle details fields')
        _require(isinstance(details['title'], str) and bool(details['title'].strip()),
                 'lifecycle title required')
        _require(isinstance(details.get('limitations', []), list) and all(
            isinstance(v, str) and bool(v.strip()) for v in details.get('limitations', [])),
            'limitations must be a text list')
        _require(isinstance(details['sources'], list), 'sources must be a list')
        declared = {(e['file'], e['sha256']) for e in entity['evidence']}
        for source in details['sources']:
            self.resolve_source(source)
            if 'file' in source:
                _require((source['file'], source['sha256']) in declared,
                         'detail source must be declared as entity evidence')
        if entity['kind'] == 'case':
            request = details['request']
            _require(isinstance(request, dict) and set(request) == {'hardware', 'model', 'framework', 'task', 'goal'},
                     'case request requires hardware, model, framework, task and goal')
            _require(all(v is None or isinstance(v, str) and bool(v.strip()) for v in request.values()),
                     'request values must be text or null')
            _require(request['task'] in (None, 'inference', 'sft', 'rl'), 'invalid case task')
            if 'category' in details:
                _require(details['category'] in ('precision-evaluation', 'repair-investigation', 'workflow-benchmark'),
                         'invalid case category')
        elif entity['kind'] == 'issue':
            _require(details['classification'] in ('confirmed-defect', 'numerical-discrepancy', 'unconfirmed-hypothesis'),
                     'invalid issue classification')
        elif entity['kind'] == 'candidate':
            _require(details['solution_type'] in ('patch', 'backport', 'upgrade', 'configuration', 'numerical-policy'),
                     'invalid solution type')
            _require(details['evidence_level'] in ('runtime-tested-in-scope', 'source-inspected'),
                     'invalid solution evidence level')
            _require(details['decision'] in ('selected-for-further-study', 'not-selected', 'not-recorded'),
                     'invalid solution decision')
            _require(isinstance(details['applicability'], str) and bool(details['applicability'].strip()),
                     'solution applicability required')
            target = details['target']
            _require(isinstance(target, dict) and set(target) == {'library', 'version', 'commit'},
                     'target requires library, version and commit')
            _require(isinstance(target['library'], str) and bool(target['library'].strip()), 'target library required')
            _require(target['version'] is None or isinstance(target['version'], str) and bool(target['version'].strip()),
                     'target version must be text or null')
            _require(target['commit'] is None or isinstance(target['commit'], str) and
                     re.fullmatch('[a-f0-9]{40}', target['commit']), 'invalid target commit')
        elif entity['kind'] == 'verification':
            _require(details['method'] in ('native-runtime', 'retained-measurements'), 'invalid verification method')
            _require(details['outcome'] in ('scoped-checks-passed', 'partial', 'mixed-results', 'not-adopted'),
                     'invalid verification outcome')
            source = details['acceptance']
            acceptance = self.resolve_source(source)
            _require(source in details['sources'], 'acceptance source must be in verification sources')
            _require(source['pointer'].rsplit('/', 1)[-1] in ('acceptance', 'numerical_acceptance'),
                     'acceptance must reference an acceptance field, not a metric')
            _require(acceptance is None or isinstance(acceptance, (str, dict)),
                     'acceptance must reference an explicit acceptance value, not a metric')

    def _relations(self, document, source_hash):
        _require(isinstance(document, dict) and set(document) == {'schema_version', 'entities', 'links'},
                 'invalid relation document fields')
        _require(type(document['schema_version']) is int and document['schema_version'] == 1,
                 'unsupported relation schema')
        _require(isinstance(document['entities'], list) and isinstance(document['links'], list),
                 'relation entities and links must be lists')
        entity_ids = set()
        for index, entity in enumerate(document['entities']):
            _require(isinstance(entity, dict) and set(entity) in (
                {'id', 'kind', 'scope', 'evidence'}, {'id', 'kind', 'scope', 'evidence', 'details'}),
                     'invalid entity fields')
            _require(isinstance(entity['kind'], str) and entity['kind'] in ENTITY_KINDS, 'invalid entity kind')
            _require(isinstance(entity['id'], str) and SLUG.fullmatch(entity['id']), 'invalid entity id')
            _require(isinstance(entity['scope'], str) and bool(entity['scope'].strip()), 'entity scope required')
            _require(entity['id'] not in entity_ids, 'duplicate entity id')
            entity_ids.add(entity['id'])
            self._evidence(entity['evidence'])
            self._details(entity)
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
            expected_left, expected_right = RELATIONS[relation]
            _require(left[0] == expected_left and (right[0] in expected_right
                     if isinstance(expected_right, set) else right[0] == expected_right),
                     'relation endpoint kind mismatch')
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
        for kind, identity in self._records:
            if kind == 'case':
                members = {right for r, left, right in edges if r == 'includes' and left == (kind, identity)}
                _require(bool(members), 'case requires explicit members')
                for r, left, right in edges:
                    if left in members and r in ('addresses', 'verifies', 'uses_workload'):
                        _require(right in members, 'case omits a member issue, solution or original workload')

    def get(self, kind, identity):
        """Return a detached record and its exact public source locator."""
        entry = deepcopy(self._records[(kind, identity)])
        content = self._content.get((kind, identity)) if hasattr(self, '_content') else None
        if content:
            entry['content'] = deepcopy(content)
        return entry

    def index(self):
        """Return stable typed identities and locators, without rewriting records."""
        return [dict(kind=k, id=i, source=deepcopy(v['source']))
                for (k, i), v in self._records.items()]

    def related(self, kind, identity):
        """Return only explicitly recorded incident links; no heuristic joins."""
        _require((kind, identity) in self._records, 'unknown record')
        endpoint = dict(kind=kind, id=identity)
        return deepcopy([link for link in self._links if endpoint in (link['from'], link['to'])])

    def case_bundle(self, identity):
        """Return explicit case members and their links; no inferred joins or pass status."""
        case = self.get('case', identity)
        endpoint = dict(kind='case', id=identity)
        includes = [link for link in self._links if link['relation'] == 'includes' and link['from'] == endpoint]
        members = {(link['to']['kind'], link['to']['id']) for link in includes}
        selected = members | {('case', identity)}
        return dict(case=case, records=[self.get(*key) for key in sorted(members)],
                    links=deepcopy([link for link in self._links
                                    if (link['from']['kind'], link['from']['id']) in selected
                                    and (link['to']['kind'], link['to']['id']) in selected]))

    def verification_bundle(self, identity):
        """Resolve the original measurements and acceptance for one verification."""
        entry = self.get('verification', identity)
        details = entry['record']['details']
        acceptance = details['acceptance']
        return dict(verification=entry, links=self.related('verification', identity),
                    resolved_sources=[dict(source=source, value=self.resolve_source(source))
                                      for source in details['sources']],
                    recorded_acceptance=self.resolve_source(acceptance) if acceptance is not None else None)

    def _case_members(self, identity):
        return {(link['to']['kind'], link['to']['id']) for link in self._links
                if link['relation'] == 'includes' and link['from'] == dict(kind='case', id=identity)}

    def case_category(self, identity):
        """Return explicit category, or a legacy category from declared member types."""
        details = self.get('case', identity)['record']['details']
        if 'category' in details:
            return details['category']
        kinds = {kind for kind, _ in self._case_members(identity)}
        return 'repair-investigation' if 'repair' in kinds else (
            'precision-evaluation' if 'comparison' in kinds else None)

    def find_cases(self, *, hardware=None, library=None, model=None, model_size=None, task=None, category=None):
        """Find cases through actual member configurations, with exact same-side matching.

        A case request is a summary, not proof of runtime identity. Configuration
        filters match one existing comparison side or repair environment together;
        unknown tasks and undeclared relationships are not filled by inference.
        """
        result = []
        for kind, identity in self._records:
            if kind != 'case' or category is not None and self.case_category(identity) != category:
                continue
            entry = self.get(kind, identity)
            request = entry['record']['details']['request']
            if all(v is None for v in (hardware, library, model, model_size, task)):
                result.append(entry)
                continue
            for member in self._case_members(identity):
                if member[0] not in ('comparison', 'repair'):
                    continue
                record = self._records[member]['record']
                if not _matches_model(record, model, model_size):
                    continue
                if task is not None:
                    declared_tasks = set(record.get('tasks') or [])
                    if record.get('task') is not None:
                        declared_tasks.add(record['task'])
                    if (task not in declared_tasks if declared_tasks else task != request['task']):
                        continue
                configurations = record.get('sides', []) if member[0] == 'comparison' else [record.get('environment', {})]
                if any((hardware is None or config.get('hardware') == hardware) and
                       (library is None or any(all(value.get(k) == v for k, v in library.items())
                                               for value in config.get('libraries', [])))
                       for config in configurations):
                    result.append(entry)
                    break
        return result

    def inventory(self):
        """Summarize declared coverage and measurement sources, without counting views as experiments."""
        kinds = Counter(kind for kind, _ in self._records)
        categories = Counter()
        covered = set()
        for kind, identity in self._records:
            if kind == 'case':
                categories[self.case_category(identity) or 'unspecified'] += 1
                covered.update(self._case_members(identity))
        coverage = {}
        for kind in ('comparison', 'repair'):
            identities = {identity for k, identity in self._records if k == kind}
            missing = sorted(identity for identity in identities if (kind, identity) not in covered)
            coverage[kind] = dict(total=len(identities), case_indexed=len(identities)-len(missing), missing=missing)
        sources = Counter()
        for (kind, _), entry in self._records.items():
            if kind == 'measurement':
                sources[entry['record'].get('source', {}).get('id') or 'unspecified'] += 1
        derived = {link['from']['id'] for link in self._links
                   if link['relation'] == 'derived_from'}
        measured = {identity for kind, identity in self._records if kind == 'measurement'}
        return dict(record_counts=dict(kinds), case_categories=dict(categories), case_coverage=coverage,
                    measurement_sources=dict(sorted(sources.items())),
                    measurement_associations=dict(total=len(measured), explicitly_derived=len(derived),
                                                  source_indexed_without_parent=len(measured-derived)))

    def search(self, request, **options):
        """Normalized demand retrieval with scoped evidence and explicit version alternatives."""
        from .search import compile_index
        from tools.community_search import search
        if not hasattr(self, '_search_index'):
            self._search_index = compile_index(self)
        return search(self._search_index, request, **options)

    def find_comparisons(self, *, hardware=None, library=None, model=None, model_size=None, task=None):
        """Exact hardware/library matching on one side; library may include version."""
        result = []
        for (kind, identity), entry in self._records.items():
            if kind != 'comparison':
                continue
            record = entry['record']
            if not _matches_model(record, model, model_size):
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
