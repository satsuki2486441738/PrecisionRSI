"""Validate or index public records without research agents or native execution."""
import argparse
import json
from pathlib import Path
from .catalog import Catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'index'))
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    parser.add_argument('--output', type=Path, help='new index file; otherwise print JSON')
    args = parser.parse_args()
    catalog = Catalog(args.site)
    if args.command == 'validate':
        if args.output is not None:
            parser.error('--output is only available for index')
        from .validation import validate, validate_evaluations
        from .differences import validate as differences
        repairs = validate(args.site)
        evaluations = validate_evaluations(args.site)
        views, metrics = differences(args.site)
        print(f'Validated {evaluations} evaluations, {repairs} repairs, {views} views and explicit relations.')
    else:
        text = json.dumps(dict(schema_version=1, records=catalog.index()), ensure_ascii=False, indent=2) + '\n'
        if args.output is None:
            print(text, end='')
        else:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)


if __name__ == '__main__':
    main()
