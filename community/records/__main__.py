"""Validate, index or inspect public cases without agents or native execution."""
import argparse
import json
from pathlib import Path
from .catalog import Catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'index', 'case', 'cases', 'search', 'verification', 'inventory'))
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    parser.add_argument('--output', type=Path, help='new query JSON file; otherwise print JSON (unavailable for validate)')
    parser.add_argument('--id', help='exact case or verification identity')
    parser.add_argument('--hardware', help='hardware label; search resolves indexed platform aliases')
    parser.add_argument('--model', help='architecture or exact recorded checkpoint name')
    parser.add_argument('--model-size', help='parameter scale (for example 0.5B), tiny, or unknown; requires --model')
    parser.add_argument('--task', help='inference/sft/rl; search also resolves indexed task aliases')
    parser.add_argument('--library', help='recorded library; search resolves indexed aliases')
    parser.add_argument('--version', help='exact library version; requires --library')
    parser.add_argument('--category', choices=('precision-evaluation', 'repair-investigation', 'workflow-benchmark'))
    parser.add_argument('--query', help='case/issue/solution keywords (search only)')
    parser.add_argument('--limit', type=int, help='1–100 cases per page (search only)')
    parser.add_argument('--offset', type=int, help='result offset (search only)')
    parser.add_argument('--include-workflows', action='store_true', help='include agent/skill benchmarks (search only)')
    parser.add_argument('--allow-alternatives', action='store_true', help='allow an explicitly different library version (search only)')
    args = parser.parse_args()
    exact = args.command in ('case', 'verification')
    if exact and not args.id:
        parser.error(args.command + ' requires --id')
    if not exact and args.id is not None:
        parser.error('--id is only available for case and verification')
    if args.command not in ('cases', 'search') and any(v is not None for v in
            (args.hardware, args.model, args.model_size, args.task, args.library, args.version, args.category)):
        parser.error('configuration/category filters are only available for cases')
    if args.command != 'cases' and args.category is not None:
        parser.error('--category is only available for cases')
    if args.command != 'search' and (any(v is not None for v in (args.query, args.limit, args.offset)) or
                                    args.include_workflows or args.allow_alternatives):
        parser.error('query/pagination/recommendation options are only available for search')
    if args.version is not None and args.library is None:
        parser.error('--version requires --library')
    if args.model_size is not None and args.model is None:
        parser.error('--model-size requires --model')
    if args.command == 'cases' and args.task is not None and args.task not in ('inference', 'sft', 'rl'):
        parser.error('--task must be inference, sft or rl')
    catalog = Catalog(args.site)
    if args.command == 'validate':
        if args.output is not None:
            parser.error('--output is unavailable for validate')
        from .validation import validate, validate_evaluations
        from .differences import validate as differences
        repairs = validate(args.site)
        evaluations = validate_evaluations(args.site)
        views, metrics = differences(args.site)
        snapshot = args.site / 'data/search-index.json'
        if snapshot.exists():
            from .search import validate_snapshot
            validate_snapshot(catalog, snapshot)
        print(f'Validated {evaluations} evaluations, {repairs} repairs, {views} views and explicit relations.')
    else:
        if args.command == 'case':
            try:
                document = dict(schema_version=1, **catalog.case_bundle(args.id))
            except KeyError:
                parser.error('unknown case: ' + args.id)
        elif args.command == 'verification':
            try:
                document = dict(schema_version=1, **catalog.verification_bundle(args.id))
            except KeyError:
                parser.error('unknown verification: ' + args.id)
        elif args.command == 'cases':
            library = dict(name=args.library) if args.library is not None else None
            if args.version is not None:
                library['version'] = args.version
            document = dict(schema_version=1, cases=catalog.find_cases(hardware=args.hardware,
                library=library, model=args.model, model_size=args.model_size, task=args.task, category=args.category))
        elif args.command == 'search':
            try:
                document = catalog.search({f: getattr(args, f) for f in
                    ('hardware', 'model', 'model_size', 'library', 'version', 'task', 'query')},
                    limit=args.limit if args.limit is not None else 20,
                    offset=args.offset if args.offset is not None else 0,
                    include_workflows=args.include_workflows, allow_alternatives=args.allow_alternatives)
            except ValueError as error:
                parser.error(str(error))
        elif args.command == 'inventory':
            document = dict(schema_version=1, **catalog.inventory())
        else:
            document = dict(schema_version=1, records=catalog.index())
        text = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
        if args.output is None:
            print(text, end='')
        else:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)


if __name__ == '__main__':
    main()
