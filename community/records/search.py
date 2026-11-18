"""Compile compact, source-bound search metadata; preserve original catalogs and evidence."""
from copy import deepcopy
import hashlib
import json
import re

from tools.community_search import hardware_platforms
from .catalog import _read
from .content import localized
from .models import model_facets, model_group

HARDWARE_RULES = [
    dict(pattern=r'ascend\s*910c|ascend910_9382|\b910_9382\b|\b910c\b', name='Ascend 910C', family='npu'),
    dict(pattern=r'ascend\s*910b', name='Ascend 910B', family='npu'),
    dict(pattern=r'ascend|npu', name='Ascend NPU', family='npu'),
    dict(pattern=r'(?<![a-z0-9_])(?:[1-9][0-9]*\s*[x×]\s*)?h(?:200|20z)(?![a-z0-9_])', name='NVIDIA H200', family='gpu'),
    dict(pattern=r'(?<![a-z0-9_])(?:[1-9][0-9]*\s*[x×]\s*)?h100(?![a-z0-9_])', name='NVIDIA H100', family='gpu'),
    dict(pattern=r'5090', name='RTX 5090', family='gpu'),
    dict(pattern=r'5060', name='RTX 5060', family='gpu'),
    dict(pattern=r'nvidia|cuda|geforce', name='NVIDIA GPU', family='gpu'),
    dict(pattern=r'cpu|aarch64|x86_64|macos|wsl', name='CPU', family='cpu'),
]
LIBRARIES = {
    'PyTorch': ['torch', 'pytorch', 'PyTorch (CUDA)'],
    'torch_npu': ['torch-npu', 'TorchNPU', 'torch_npu'],
    'vLLM': ['vllm'], 'vllm_ascend': ['vllm-ascend', 'vllm_ascend', 'vLLM-Ascend selected source'],
    'Transformers': ['transformers', 'huggingface transformers'], 'PEFT': ['peft'], 'TRL': ['trl'],
    'ms-swift': ['ms_swift', 'ms-swift'], 'LLaMA-Factory': ['llamafactory', 'llama-factory', 'llama_factory'],
    'verl': ['verl', 'VeRL'], 'OpenRLHF': ['openrlhf'], 'SGLang': ['sglang'],
    'NeMo RL': ['nemo-reinforcer'],
    'Hugging Face Hub': ['huggingface-hub', 'huggingface_hub'],
    'Megatron Bridge': ['megatron-bridge'], 'Megatron Core': ['megatron-core'],
    'Transformer Engine': ['transformer-engine'], 'CUDA runtime': ['cuda_runtime'],
}
OFFICIAL_MODELS = ('qwen/', 'deepseek-ai/', 'meta-llama/', 'google/', 'openai/', 'nvidia/')


def _key(value):
    return ' '.join(str(value).strip().lower().split())


def _library_names(rows):
    names = {}
    for (kind, _), entry in rows.items():
        record = entry['record']
        sides = record['sides'] if kind == 'comparison' else [record['environment']] if kind == 'repair' else []
        for side in sides:
            for lib in side.get('libraries', []):
                name = ' '.join(lib['name'].split())
                names.setdefault(_key(name), set()).add(name)
    # Keep observed branding for undeclared libraries; choose independently of record order.
    canonical = {key: min(labels, key=lambda label: (label == key, label)) for key, labels in names.items()}
    canonical.update({_key(alias): name for name, aliases in LIBRARIES.items() for alias in (name, *aliases)})
    return canonical


def _versioned(library):
    version = library.get('version')
    return (bool(re.fullmatch('[a-fA-F0-9]{40}', library.get('commit') or '')) or
            isinstance(version, str) and bool(version.strip()) and _key(version) not in
            {'unknown', 'not-recorded', 'not recorded', 'unspecified', 'latest', 'main', 'master', 'head', 'selected source', '未记录', '未知'})


def _framework_annotations(site, rows):
    path = site / 'data/frameworks.json'
    if not path.exists():
        return {}, None
    data, digest = _read(path)
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'evaluations', 'repairs'} or
            type(data['schema_version']) is not int or data['schema_version'] != 1 or
            not isinstance(data['evaluations'], dict) or not isinstance(data['repairs'], dict)):
        raise ValueError('Invalid legacy framework annotations')
    annotations = {}
    for identity, entries in data['evaluations'].items():
        source = rows.get(('comparison', identity))
        if source is None or not isinstance(entries, dict) or not entries:
            raise ValueError('Invalid framework annotation for ' + identity)
        sides = {side['id'] for side in source['record']['sides']}
        for name, selected in entries.items():
            if not name.strip() or selected is not None and (
                    not isinstance(selected, list) or not selected or
                    not all(isinstance(side, str) and side in sides for side in selected) or
                    len(selected) != len(set(selected))):
                raise ValueError('Invalid framework sides for ' + identity + ': ' + name)
            for side in sides if selected is None else selected:
                annotations.setdefault(('comparison', identity, side), []).append(name)
    for identity, names in data['repairs'].items():
        if ('repair', identity) not in rows or not isinstance(names, list) or not names:
            raise ValueError('Invalid repair framework annotation for ' + identity)
        annotations['repair', identity, None] = names
    return annotations, digest


def _frameworks(side, record, legacy, library_names):
    if legacy is not None and 'frameworks' in side:
        raise ValueError('Frameworks declared twice for ' + record['id'])
    affected = record.get('affected_library')
    names = side.get('frameworks', legacy if legacy is not None else [affected] if affected else [])
    if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError('Expected a list of framework names for ' + record['id'])
    canonical = lambda name: library_names.get(_key(name), name)
    recorded = {canonical(lib['name']) for lib in side.get('libraries', [])}
    names = [canonical(name) for name in names]
    if len(names) != len(set(names)) or not set(names) <= recorded:
        raise ValueError('Frameworks must name distinct libraries on the same side of ' + record['id'])
    if affected and canonical(affected) not in names:
        raise ValueError('Frameworks must include the affected library for ' + record['id'])
    return sorted(names)


def _complete(coverage):
    rows = coverage if isinstance(coverage, list) else [coverage]
    return (bool(rows) and any(type(r.get('expected')) in (int, float) and r['expected'] > 0 for r in rows) and
            all(type(r.get('observed')) in (int, float) and type(r.get('expected')) in (int, float)
                and r['expected'] >= 0 and r['observed'] == r['expected'] for r in rows))


def compile_index(catalog):
    """Build metadata from explicit members and same-runtime verification links only."""
    rows = { (r['kind'], r['id']): catalog.get(r['kind'], r['id']) for r in catalog.index()
             if r['kind'] != 'measurement'}
    library_names = _library_names(rows)
    annotations, annotations_digest = _framework_annotations(catalog.site, rows)
    index = dict(schema_version=1, kind='numericbridge-search-index', policy_version=2,
                 observation_date='', sources=[], hardware_rules=deepcopy(HARDWARE_RULES),
                 aliases={f: {} for f in ('hardware', 'model', 'model_size', 'library', 'task')}, cases=[],
                 content={f'{kind}:{identity}': catalog.content(kind, identity) for kind, identity in rows
                          if catalog.content(kind, identity)})
    alias_sets = {field: {} for field in ('hardware', 'model', 'model_size', 'library')}
    def alias(field, raw, name):
        alias_sets[field].setdefault(_key(raw), set()).add(name)
    def entity(kind, identity):
        row = rows[(kind, identity)]
        content = catalog.content(kind, identity)
        details = row['record']['details']
        return dict(id=identity, title=localized(content, 'title', default=details['title']),
                    summary=localized(content, 'summary'), content=content,
                    **{key: deepcopy(details[key]) for key in
                       ('classification', 'solution_type', 'target', 'evidence_level', 'decision') if key in details},
                    source=deepcopy(row['source']), evidence=deepcopy(row['record']['evidence']))
    for entry in rows.values():
        if entry['kind'] != 'case':
            continue
        row = entry['record']
        bundle = catalog.case_bundle(row['id'])
        members = {(r['kind'], r['id']) for r in bundle['records']}
        links = bundle['links']
        content = catalog.content('case', row['id'])
        case = dict(id=row['id'], title=localized(content, 'title', default=row['details']['title']),
                    scope=localized(content, 'summary'), content=content,
                    category=catalog.case_category(row['id']), source=deepcopy(entry['source']),
                    evidence=deepcopy(row['evidence']), configurations=[], issues=[], solutions=[])
        for kind, identity in sorted(members):
            if kind == 'candidate':
                solution = entity(kind, identity)
                solution['issue_ids'] = [link['to']['id'] for link in links if link['relation'] == 'addresses'
                                         and link['from'] == dict(kind=kind, id=identity)]
                solution['verification_ids'] = [link['from']['id'] for link in links if link['relation'] == 'verifies'
                                                and link['to'] == dict(kind=kind, id=identity)]
                case['solutions'].append(solution)
            elif kind == 'issue':
                issue = entity(kind, identity)
                issue['record_ids'] = [link['to']['id'] for link in links if link['relation'] == 'documents'
                                       and link['from'] == dict(kind='issue', id=identity) and link['to']['kind'] == 'repair']
                case['issues'].append(issue)
            if kind not in ('comparison', 'repair'):
                continue
            record_entry = rows[(kind, identity)]
            record = record_entry['record']
            for position, side in enumerate(record['sides'] if kind == 'comparison' else [record['environment']]):
                model = record.get('model', {})
                name = model.get('name') or ''
                architecture, size = model_facets(model)
                group = model_group(model, architecture)
                model_keys = {_key(name)}
                if name.lower().startswith(OFFICIAL_MODELS):
                    model_keys.add(_key(name.split('/', 1)[1]))
                elif name.lower().startswith('qwen'):
                    model_keys.add(_key('Qwen/' + name))
                for value in model_keys:
                    alias('model', value, name)
                identity_keys = model_keys - {_key(architecture), _key(group)}
                for label in (architecture, group):
                    model_keys.add(_key(label))
                    alias('model', label, label)
                if architecture == 'Qwen2':
                    model_keys.add('qwen2.5')
                    alias('model', 'Qwen2.5', architecture)
                alias('model_size', size, size)
                platforms = hardware_platforms(side['hardware'], index)
                for platform in platforms:
                    alias('hardware', platform, platform)
                libraries = []
                for lib in side.get('libraries', []):
                    canonical = library_names[_key(lib['name'])]
                    keys = sorted({_key(lib['name']), _key(canonical)})
                    for value in [lib['name'], canonical, *LIBRARIES.get(canonical, [])]:
                        alias('library', value, canonical)
                    libraries.append(dict(deepcopy(lib), keys=keys))
                annotation = annotations.get((kind, identity, side.get('id') if kind == 'comparison' else None))
                frameworks = _frameworks(side, record, annotation, library_names)
                tasks = record.get('tasks') or ([record['task']] if record.get('task') else
                                                [row['details']['request']['task']] if row['details']['request']['task'] else [])
                comparisons = [r for r in record.get('comparisons', []) if side.get('id') in r.get('side_ids', [])]
                validation = record.get('validation', {})
                coverage = [count for r in comparisons for count in r['coverage']] if kind == 'comparison' else validation.get('coverage', {})
                description = catalog.content(kind, identity)
                config = dict(id=f'{kind}:{identity}:{side.get("id", position)}', hardware=side['hardware'],
                              platforms=platforms, model=name, model_revision=model.get('revision'),
                              model_architecture=architecture, model_group=group, model_size=size,
                              frameworks=frameworks, libraries=libraries,
                              tasks=tasks, settings=deepcopy(side.get('settings', {})),
                              scope=localized(description, 'summary'), content=description,
                              date=record.get('date'), record=dict(kind=kind, id=identity, source=deepcopy(record_entry['source']),
                                                               status=record.get('status'), acceptance=record.get('acceptance') if kind == 'comparison' else validation.get('acceptance')),
                              coverage=deepcopy(coverage), recommendations=[], recommendation_gaps=[],
                              keys=dict(hardware=[_key(p) for p in platforms], model=sorted(model_keys),
                                        model_identity=sorted(identity_keys),
                                        library=sorted({_key(name) for name in frameworks})))
                gaps = config['recommendation_gaps']
                if not frameworks:
                    gaps.append('no-recorded-framework')
                if len(platforms) != 1:
                    gaps.append('multi-platform-environment-not-isolated')
                if any(p in ('Ascend NPU', 'NVIDIA GPU') for p in platforms):
                    gaps.append('hardware-model-not-recorded')
                if not libraries or not all(_versioned(lib) for lib in libraries):
                    gaps.append('missing-pinned-dependency-version')
                if not _complete(coverage):
                    gaps.append('incomplete-measured-coverage')
                if case['category'] == 'workflow-benchmark':
                    gaps.append('workflow-only-evidence')
                if kind == 'comparison':
                    if record.get('acceptance') != 'passed':
                        gaps.append('no-recorded-numerical-acceptance')
                    if record.get('status') != 'completed' or record.get('scope_status') == 'partial' or record.get('whole_task_completed') is False:
                        gaps.append('incomplete-workload')
                    if not comparisons or not all(r.get('metrics') for r in comparisons):
                        gaps.append('missing-numerical-measurements')
                    if not gaps:
                        config['recommendations'].append(dict(basis='recorded-numerical-acceptance', candidate_id=None,
                            verification_id=None, workload_id=None, scope=config['scope']))
                else:
                    if record.get('status') != 'validated' or not validation.get('metrics') or not validation.get('checks'):
                        gaps.append('missing-original-workload-checks')
                    for candidate_kind, candidate_id in sorted(members):
                        if candidate_kind != 'candidate' or gaps:
                            continue
                        solution = rows[(candidate_kind, candidate_id)]['record']['details']
                        target = solution['target']
                        if solution['evidence_level'] != 'runtime-tested-in-scope' or solution['decision'] != 'not-recorded':
                            continue
                        target_key = _key(library_names.get(_key(target['library']), target['library']))
                        if target_key not in config['keys']['library']:
                            continue
                        lib = next((l for l in libraries if target_key in l['keys'] and
                                    (not target.get('version') or target['version'] == l.get('version')) and
                                    (not target.get('commit') or target['commit'] == l.get('commit'))), None)
                        if lib is None:
                            continue
                        for vk, vid in sorted(members):
                            if vk != 'verification':
                                continue
                            v = rows[(vk, vid)]['record']
                            if v['details']['outcome'] != 'scoped-checks-passed' or not any(
                                    s.get('kind') == 'repair' and s.get('id') == identity and s.get('pointer') == '/validation/metrics'
                                    for s in v['details']['sources']):
                                continue
                            if not any(l['relation'] == 'verifies' and l['from'] == dict(kind=vk, id=vid)
                                       and l['to'] == dict(kind='candidate', id=candidate_id) for l in links):
                                continue
                            workload = next((l['to']['id'] for l in links if l['relation'] == 'uses_workload' and
                                l['from'] == dict(kind=vk, id=vid) and l['to']['kind'] == 'workload' and
                                any(s.get('kind') == 'repair' and s.get('id') == identity and s.get('pointer') == '/environment'
                                    for s in rows[('workload', l['to']['id'])]['record']['details']['sources'])), None)
                            if workload:
                                config['recommendations'].append(dict(basis='scoped-original-workload-repair',
                                    candidate_id=candidate_id, verification_id=vid, workload_id=workload,
                                    library_keys=lib['keys'], scope=localized(catalog.content('candidate', candidate_id), 'summary')))
                    if not config['recommendations']:
                        gaps.append('no-supported-original-workload-solution')
                case['configurations'].append(config)
        texts = [case['id'], case['title'], case['scope'], json.dumps(content, ensure_ascii=False),
                 *[s['title'] for s in case['solutions']], *[i['title'] for i in case['issues']]]
        for config in case['configurations']:
            texts.extend([config['model'], config['model_architecture'], config['model_group'], config['model_size'], config['hardware'], *config['tasks']])
            texts.extend(lib['name'] + ' ' + (lib.get('version') or '') for lib in config['libraries'])
        for kind, identity in sorted(members):
            if kind in ('comparison', 'repair'):
                observation = rows[(kind, identity)]['record']
                texts.extend([observation.get('title', ''), observation.get('summary', '')])
                for comparison in observation.get('comparisons', []):
                    texts.extend([comparison['title'], comparison['scope']])
                    texts.extend(metric['label'] for metric in comparison['metrics'])
        case['search_text'] = _key(' '.join(texts))
        index['cases'].append(case)
    for field, aliases in alias_sets.items():
        index['aliases'][field] = {key: next(iter(names)) for key, names in sorted(aliases.items()) if len(names) == 1}
    index['aliases']['task'] = {'推理': 'inference', 'infer': 'inference', 'inference': 'inference',
                                '监督微调': 'sft', 'sft': 'sft', 'rl': 'rl', '强化学习': 'rl'}
    for file in ('evaluations.json', 'repairs.json', 'relations.json'):
        raw = (catalog.site / 'data' / file).read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        expected = {entry['source']['sha256'] for entry in rows.values() if entry['source']['file'] == 'data/' + file}
        if expected and expected != {source_hash}:
            raise ValueError('Catalog changed during search compilation: ' + file)
        document = json.loads(raw)
        index['sources'].append(dict(file='data/' + file, sha256=source_hash))
        index['observation_date'] = max(index['observation_date'], document.get('updated_at', '')[:10])
    if catalog._content_hash is not None:
        raw = (catalog.site / 'data/content.json').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != catalog._content_hash:
            raise ValueError('Community content changed during search compilation')
        index['sources'].append(dict(file='data/content.json', sha256=digest))
    if annotations_digest is not None:
        index['sources'].append(dict(file='data/frameworks.json', sha256=annotations_digest))
    index['cases'].sort(key=lambda c: c['id'])
    return index


def validate_snapshot(catalog, path):
    """Reject stale or edited metadata rather than silently serving different eligibility rules."""
    expected = json.dumps(compile_index(catalog), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    if path.read_text(encoding='utf-8') != expected:
        raise ValueError('Community search index is stale or edited; run python -B -m scripts export-search --refresh')
