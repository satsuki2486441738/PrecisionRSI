"""Expose the public NumericBridge search index as a read-only MCP tool."""

import argparse
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field, StringConstraints

from tools.community_search import load_index, search


DEFAULT_INDEX = Path(__file__).resolve().parent / 'site' / 'data' / 'search-index.json'
LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')
LookupText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                              min_length=1, max_length=1000)]
PageSize = Annotated[int, Field(strict=True, ge=1, le=20)]
PageOffset = Annotated[int, Field(strict=True, ge=0, le=100000)]


def create_server(index_path: Path = DEFAULT_INDEX, *, index_sha256: str | None = None) -> MCPServer:
    """Load one reviewed snapshot at startup; tool callers cannot select its source."""
    index, receipt = load_index(index_path, index_sha256)
    index_identity = {key: receipt[key] for key in ('index_sha256', 'bytes')}
    server = MCPServer('numericbridge-community', version='1.0.0', subscriptions=False,
                       instructions='Search public NumericBridge precision evaluations and repair evidence.')

    @server.tool(
        title='Search NumericBridge community',
        annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def search_community(
        hardware: LookupText,
        model: LookupText,
        library: LookupText,
        task: Literal['inference', 'sft', 'rl'],
        version: LookupText | None = None,
        model_size: LookupText | None = None,
        limit: PageSize = 20,
        offset: PageOffset = 0,
    ) -> dict[str, Any]:
        """Search by target hardware, model, tested framework (library), and task.

        Pass version only for a fixed framework version and model_size when
        needed. Returns matching cases, issues, solutions and verification references,
        supported stack recommendations, and the index identity. No match is
        a normal result. Listed alternatives never replace a fixed version.
        """
        request = dict(hardware=hardware, model=model, library=library, task=task,
                       version=version, model_size=model_size)
        result = search(index, request, limit=limit, offset=offset)
        result['retrieval'] = index_identity.copy()
        return result

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=('streamable-http', 'stdio'), default='streamable-http')
    parser.add_argument('--host', default='127.0.0.1', help='HTTP bind address')
    parser.add_argument('--port', type=int, default=8769, help='HTTP port')
    parser.add_argument('--allow-host', action='append', default=[], help='allowed HTTP Host header; repeatable')
    parser.add_argument('--allow-origin', action='append', default=[], help='allowed HTTP Origin header; repeatable')
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX, help='server-local public index snapshot')
    parser.add_argument('--sha256', help='expected SHA-256 of the index snapshot')
    args = parser.parse_args()
    if args.transport == 'streamable-http':
        if not 1 <= args.port <= 65535:
            parser.error('--port must be between 1 and 65535')
        if args.host not in LOOPBACK_HOSTS and not args.allow_host:
            parser.error('binding outside loopback requires an explicit --allow-host')
    try:
        server = create_server(args.index, index_sha256=args.sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    if args.transport == 'stdio':
        server.run('stdio')
        return

    local_hosts = ['127.0.0.1', 'localhost', '[::1]']
    security = TransportSecuritySettings(
        allowed_hosts=[*local_hosts, *(f'{name}:{args.port}' for name in local_hosts), *args.allow_host],
        allowed_origins=args.allow_origin,
    )
    server.run('streamable-http', host=args.host, port=args.port, streamable_http_path='/mcp',
               stateless_http=True, json_response=True, max_request_body_size=64 * 1024,
               transport_security=security)


if __name__ == '__main__':
    main()
