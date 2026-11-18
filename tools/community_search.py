"""Search a pinned community metadata index without model execution (stdlib).
Call: python scripts/community_search.py --hardware H200 --model MODEL --library LIB
Input: --index local JSON / explicit HTTPS URL, optional --sha256; demand flags or --request JSON.
Output: normalized demand, matching cases, issues, solutions and supported version combinations.
Requires: Standard library. An exported skill includes a dated, offline metadata snapshot.

Details:
Library matches the tested framework or repair target. Dependencies remain in each stack.
Version is optional. Alternatives list other recorded versions of the requested
configuration; --allow-alternatives permits selecting one. Empty matches supply
no community context. Results include source hashes and the loaded index SHA.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

__all__ = ['hardware_platforms', 'normalize_request', 'search', 'load_index']
FIELDS = ('hardware', 'model', 'library', 'task', 'version', 'model_size')
CORE = ('hardware', 'model', 'library')
MAX_BYTES = 16 * 1024 * 1024


def _key(value):
    return ' '.join(str(value).strip().lower().split())


def hardware_platforms(value, index):
    """Apply the index's explicit SKU rules, retaining unknown names and all named accelerators."""
    text = str(value).strip()
    hardware = re.sub(r'\bno\s+(?:model\s+or\s+)?(?:accelerators?|gpus?|npus?)\b', ' ', text.lower())
    found = [(r['name'], r['family']) for r in index['hardware_rules']
             if re.search(r['pattern'], hardware)]
    names = list(dict.fromkeys(name for name, _ in found))
    if any(name in names for name in ('Ascend 910C', 'Ascend 910B')):
        names = [name for name in names if name != 'Ascend NPU']
    if any(name.startswith(('NVIDIA H', 'RTX ')) for name in names):
        names = [name for name in names if name != 'NVIDIA GPU']
    if any(name != 'CPU' for name in names):
        names = [name for name in names if name != 'CPU']
    if names == ['NVIDIA GPU'] and re.search(r'(?:nvidia|geforce)\s*(?:[a-z]+\s*)?\d', hardware):
        return [text]
    if names == ['CPU'] and re.search(r'offload|gpu|npu|accelerator', hardware):
        return [text]
    return names or [text]


def _valid_index(index):
    if (not isinstance(index, dict) or type(index.get('schema_version')) is not int or index.get('schema_version') != 1 or
            index.get('kind') != 'numericbridge-search-index' or type(index.get('policy_version')) is not int or index.get('policy_version') != 2 or
            not isinstance(index.get('cases'), list) or not isinstance(index.get('aliases'), dict) or
            not isinstance(index.get('sources'), list) or not isinstance(index.get('hardware_rules'), list)):
        raise ValueError('Unsupported community search index')
    for source in index['sources']:
        if (not isinstance(source, dict) or set(source) != {'file', 'sha256'} or
                source.get('file') not in ('data/evaluations.json', 'data/repairs.json', 'data/relations.json',
                                           'data/content.json', 'data/frameworks.json') or
                not isinstance(source.get('sha256'), str) or
                not re.fullmatch('[a-f0-9]{64}', source.get('sha256', ''))):
            raise ValueError('Invalid source identity in search metadata')
    files = [s['file'] for s in index['sources']]
    if (len(files) != len(set(files)) or not
            {'data/evaluations.json', 'data/repairs.json', 'data/relations.json'} <= set(files)):
        raise ValueError('Search metadata must pin distinct source catalogs')
    for rule in index['hardware_rules']:
        try:
            re.compile(rule['pattern'])
        except (KeyError, TypeError, re.error) as error:
            raise ValueError('Invalid hardware rule') from error
    ids = set()
    for case in index['cases']:
        if not isinstance(case.get('id'), str) or case['id'] in ids or not isinstance(case.get('configurations'), list):
            raise ValueError('Invalid or duplicate search case')
        ids.add(case['id'])


def normalize_request(request, index):
    """Validate fields and resolve only explicit, unambiguous indexed aliases."""
    if not isinstance(request, dict) or set(request) - set(FIELDS) - {'query'}:
        raise ValueError('Request supports hardware, model, library, task, version, model_size and query only')
    values = {}
    for field in (*FIELDS, 'query'):
        value = request.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(field + ' must be text or null')
        value = (value or '').strip()
        if len(value) > 1000:
            raise ValueError(field + ' is too long')
        values[field] = index['aliases'].get(field, {}).get(_key(value), value) if value else ''
    if values['hardware']:
        platforms = hardware_platforms(values['hardware'], index)
        if len(platforms) != 1:
            raise ValueError('Choose one target hardware platform; cross-platform evidence remains in results')
        values['hardware'] = platforms[0]
    if values['version'] and not values['library']:
        raise ValueError('version requires library')
    if values['model_size'] and not values['model']:
        raise ValueError('model_size requires model')
    if values['task'] and values['task'] not in ('inference', 'sft', 'rl'):
        raise ValueError('task must be inference, sft or rl')
    return values


def _differences(config, request, *, ignore_version=False):
    differences = []
    for field in FIELDS:
        value = request[field]
        if not value or ignore_version and field == 'version':
            continue
        if field == 'version':
            matches = any(_key(request['library']) in lib['keys'] and lib.get('version') == value
                          for lib in config['libraries'])
        elif field == 'task':
            matches = value in config['tasks']
        elif field == 'model_size':
            matches = _key(value) == _key(config.get('model_size', ''))
        else:
            matches = _key(value) in config['keys'][field]
        if not matches:
            differences.append(field)
    return differences


def _configuration(config):
    result = {k: deepcopy(v) for k, v in config.items() if k not in ('keys', 'recommendations')}
    result['libraries'] = [{k: v for k, v in lib.items() if k != 'keys'} for lib in result['libraries']]
    return result


def _case_result(case, configurations, differences=None):
    record_ids = {c['record']['id'] for c in configurations}
    return dict(id=case['id'], title=case['title'], category=case['category'], scope=case['scope'],
                content=deepcopy(case.get('content', {})),
                source=deepcopy(case['source']), evidence=deepcopy(case['evidence']),
                configurations=[_configuration(c) for c in configurations],
                issues=deepcopy([i for i in case['issues'] if record_ids.intersection(i['record_ids'])]),
                solutions=deepcopy(case['solutions']), differences=differences or [])


def search(index, request, *, limit=20, offset=0, related_limit=0,
           include_workflows=False, allow_alternatives=False):
    """Return bounded configuration retrieval; scientific recommendation support is index-owned."""
    _valid_index(index)
    for value, lower, upper, name in ((limit, 1, 100, 'limit'), (offset, 0, 100000, 'offset'),
                                       (related_limit, 0, 20, 'related_limit')):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer between {lower} and {upper}')
    normalized = normalize_request(request, index)
    if type(include_workflows) is not bool or type(allow_alternatives) is not bool:
        raise ValueError('Workflow/alternative options must be booleans')
    missing = [f for f in CORE if not normalized[f]]
    terms = _key(normalized['query']).split()
    matches, related, choices, alternatives, gaps = [], [], [], [], set()
    configuration_count = 0
    for case in sorted(index['cases'], key=lambda c: c['id']):
        if case['category'] == 'workflow-benchmark' and not include_workflows:
            continue
        exact = [c for c in case['configurations'] if not _differences(c, normalized)]
        if exact:
            configuration_count += 1
        if not all(term in case['search_text'] for term in terms):
            continue
        if exact:
            matches.append(_case_result(case, exact))
            gaps.update(g for c in exact for g in c['recommendation_gaps'])
        if not exact and related_limit:
            ranked = [(len(diff), c['id'], diff, c) for c in case['configurations']
                      if (diff := _differences(c, normalized))]
            ranked.sort(key=lambda row: row[:2])
            if (ranked and len(ranked[0][2]) < sum(bool(normalized[f]) for f in FIELDS) and
                    sum(f in ranked[0][2] for f in CORE) <= 1):
                diff, config = ranked[0][2:]
                related.append((len(diff), case['id'], _case_result(case, [config], diff)))
        if missing or case['category'] == 'workflow-benchmark':
            continue
        for config in case['configurations']:
            if _differences(config, normalized, ignore_version=True):
                continue
            model_key = _key(normalized['model'])
            if model_key not in config['keys'].get('model_identity', [_key(config['model'])]):
                if config['recommendations']:
                    gaps.add('specific-model-required-for-recommendation')
                continue
            for support in config['recommendations']:
                if support.get('library_keys') and _key(normalized['library']) not in support['library_keys']:
                    continue
                option = dict(case_id=case['id'], configuration=_configuration(config),
                              **{k: deepcopy(v) for k, v in support.items() if k != 'library_keys'})
                option['differences'] = _differences(config, normalized)
                option['version_match'] = 'alternative' if 'version' in option['differences'] else 'exact'
                (alternatives if option['differences'] else choices).append(option)
    def order(option):
        config = option['configuration']
        return (0 if option.get('candidate_id') else 1, -(int(config['date'].replace('-', '')) if config.get('date') else 0),
                config['record']['id'], config['id'], option.get('candidate_id') or '', option['case_id'])
    def distinct(options):
        found = {}
        for option in sorted(options, key=order):
            found.setdefault((option['configuration']['id'], option.get('candidate_id')), option)
        return list(found.values())
    choices, alternatives = distinct(choices), distinct(alternatives)
    recommendations = choices or (alternatives if allow_alternatives else [])
    if missing:
        gaps.add('missing-required-input')
    elif not configuration_count:
        gaps.add('no-matching-configuration')
    elif not recommendations:
        gaps.add('insufficient-recommendation-evidence')
    status = ('needs-input' if missing else 'matched' if matches else 'no-keyword-match' if configuration_count
              else 'related-only' if related else 'no-records')
    return dict(schema_version=1, normalized_request=normalized, missing_input=missing, status=status,
                snapshot=dict(sources=deepcopy(index['sources']), policy_version=index['policy_version'],
                              observation_date=index['observation_date']),
                configuration_case_count=configuration_count, total=len(matches), offset=offset, limit=limit,
                matches=matches[offset:offset + limit],
                related=[] if matches else [row[2] for row in sorted(related, key=lambda r: r[:2])[:related_limit]],
                recommendation_status='available' if recommendations else 'needs-input' if missing else 'insufficient-evidence',
                recommendation=recommendations[0] if recommendations else None,
                recommendations=recommendations[:10], alternatives=alternatives[:10],
                recommendation_gaps=[] if recommendations else sorted(gaps))


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON value: ' + value)
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def load_index(source, expected_sha256=None):
    """Read bounded metadata from a local file or an explicit HTTPS endpoint; return digest receipt."""
    source = str(source)
    url = urlsplit(source)
    if url.scheme in ('http', 'https'):
        if url.scheme != 'https' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Remote metadata requires HTTPS')
        with urlopen(source, timeout=20) as response:
            final = urlsplit(response.geturl())
            if final.scheme != 'https' and final.hostname not in ('127.0.0.1', 'localhost', '::1'):
                raise ValueError('Metadata redirected to an unsupported endpoint')
            raw = response.read(MAX_BYTES + 1)
    elif url.scheme:
        raise ValueError('Unsupported metadata URL scheme')
    else:
        with Path(source).open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Community index exceeds 16 MiB metadata limit')
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError('Community index SHA-256 mismatch')
    index = _decode(raw)
    _valid_index(index)
    return index, dict(index_sha256=digest, bytes=len(raw), source=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:')[0])
    parser.add_argument('--index', help='metadata JSON file or explicit HTTPS URL')
    parser.add_argument('--sha256', help='expected SHA-256 of the metadata index')
    parser.add_argument('--request', type=Path, help='request JSON; cannot be combined with demand flags')
    for field in (*FIELDS, 'query'):
        parser.add_argument('--' + field.replace('_', '-'))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--related-limit', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--include-workflows', action='store_true')
    parser.add_argument('--allow-alternatives', action='store_true')
    parser.add_argument('--output', type=Path, help='new output file; existing paths are rejected')
    args = parser.parse_args()
    try:
        request = {f: getattr(args, f) for f in (*FIELDS, 'query') if getattr(args, f) is not None}
        if args.request:
            if request:
                raise ValueError('--request cannot be combined with demand flags')
            request = _decode(args.request.read_bytes())
        source = args.index
        if source is None:
            root = Path(__file__).resolve().parents[1]
            source = next((p for p in (root / 'community/site/data/search-index.json',
                                      root / 'data/community-search.json') if p.is_file()), None)
            if source is None:
                raise ValueError('Provide --index; no bundled metadata snapshot is available')
        index, receipt = load_index(source, args.sha256)
        result = search(index, request, limit=args.limit, offset=args.offset, related_limit=args.related_limit,
                        include_workflows=args.include_workflows, allow_alternatives=args.allow_alternatives)
        result['retrieval'] = receipt
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)
        else:
            print(text, end='')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
