"""Build or check the compact public community search index without network access."""
import argparse
import json
from pathlib import Path

from community.records.catalog import Catalog
from community.records.search import compile_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'community/site')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--refresh', action='store_true', help='replace the derived index explicitly')
    parser.add_argument('--check', action='store_true', help='reject a missing, stale or edited derived index')
    args = parser.parse_args()
    output = args.output or args.site / 'data/search-index.json'
    try:
        index = compile_index(Catalog(args.site))
        text = json.dumps(index, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
        if args.check:
            if args.refresh:
                raise ValueError('--check cannot be combined with --refresh')
            if output.read_text(encoding='utf-8') != text:
                raise ValueError('Community search index is stale or edited; regenerate from the reviewed catalogs')
        else:
            with output.open('w' if args.refresh else 'x', encoding='utf-8') as stream:
                stream.write(text)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(dict(cases=len(index['cases']), bytes=len(text.encode('utf-8')), checked=args.check)))


if __name__ == '__main__':
    main()
