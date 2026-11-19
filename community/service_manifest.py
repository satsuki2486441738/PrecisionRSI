"""Export public endpoint discovery metadata for the static community site."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .intake import endpoint


def descriptor(url):
    if not url:
        return {'schema_version': 1, 'submission': None}
    parts = urlsplit(endpoint(url))
    if parts.scheme != 'https' or (parts.port is not None and not 1 <= parts.port <= 65535):
        raise ValueError('The published submission endpoint must use HTTPS')
    try:
        ip = ipaddress.ip_address(parts.hostname)
    except ValueError:
        name = parts.hostname.rstrip('.')
        public = '.' in name and not name.endswith(('.localhost', '.local', '.internal', '.lan'))
    else:
        public = ip.is_global and not ip.is_multicast
    if not public:
        raise ValueError('The published submission endpoint must use a public host')
    return {'schema_version': 1, 'submission': {'url': url, 'method': 'POST', 'auth': 'bearer'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        value = descriptor(os.environ.get('NUMERICBRIDGE_INTAKE_URL', '').strip())
    except ValueError:
        parser.error('Set NUMERICBRIDGE_INTAKE_URL to a public HTTPS /v1/submissions URL without credentials or a query')
    args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
