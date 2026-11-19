# NumericBridge community

This directory owns the community website and its public record tools together.

| Directory | Purpose |
|---|---|
| [site/](site/README.md) | Static pages, public JSON records and selected evidence |
| [records/](records/README.md) | Python validation, read-only search and explicit relations |
| [mcp_search.py](mcp_search.py) | Optional MCP server for Agent access to community search |
| [mcp_submission.py](mcp_submission.py) | Local MCP for reviewing and submitting contributions |

```sh
python -B -m community.records validate
python -B -m community.records index --output /tmp/numericbridge-records.json
python -m http.server 8000 --directory community/site
```

Run commands from the repository root. GitHub Pages publishes `community/site/`
and reads the JSON directly. Deploy the MCP endpoint separately when needed.

## MCP search

Install the optional server dependencies and start the endpoint from the repository root:

```sh
python -m pip install -r community/requirements-mcp.txt
python -B -m community.mcp_search
```

Connect an MCP client using Streamable HTTP at `http://127.0.0.1:8769/mcp`.
For a client that starts its own process, use `python -B -m community.mcp_search --transport stdio`
from the repository root instead.

The server exposes one tool, `search_community`. Its required arguments are
`hardware`, `model`, `library` (the tested framework), and `task` (`inference`,
`sft`, or `rl`). Optional filters are `version` and `model_size`; `limit` and
`offset` page the case list. Example arguments:

```json
{"hardware":"Ascend 910C","model":"Qwen3.5-9B-Base","library":"MindSpeed","task":"rl"}
```

The structured result uses the shared search tool's versioned contract:
`matches` contains case, issue, solution and verification references;
`recommendation` is `null` when the evidence does not support a version
combination for the request. `alternatives` can list evidenced versions of the
same stack when the request pins another version. `no-records` is a successful
empty lookup. `retrieval.index_sha256` identifies the snapshot.
The server reads the public search index at startup;
restart after installing a new snapshot. Operators can set `--index` to a local
snapshot and `--sha256` to its expected digest.

For a public endpoint, run the service behind an HTTPS reverse proxy that
preserves the external Host header. Pass that header with `--allow-host` and
use `--allow-origin` for any required Origin values. For example, for the
placeholder public address `https://mcp.example.org/mcp`:

```sh
python -B -m community.mcp_search --allow-host mcp.example.org
```

The MCP tool reads public records. It does not accept data uploads, change
community records or run model workloads.

## MCP contribution

The [workflow](../skills/numericbridge/references/workflow.md#resume-and-deliver)
defines when to offer submission and how to review the draft. Anonymization
applies to the submitted content; GitHub still shows the PR author's account.

| Operation | Input | Result |
|---|---|---|
| `prepare_community_submission` | Required `title`. Optional `evaluations`, `repairs`, `library_groups`, `entities` and `links` are arrays in the existing record format. `evidence` maps new `evidence/submissions/*.json` paths to public JSON objects. | `ready` returns `draft_id`, `draft_sha256` and `preview` with the exact additions and destination; `invalid` returns `findings`. No data goes to GitHub. |
| `submit_community` | Only `draft_id` and `draft_sha256`. The MCP client presents a separate user confirmation. | `submitted` returns `pull_request_url` and `reused`; an accepted retry returns the existing PR. `declined`, `cancelled`, `approval_unavailable`, `invalid_draft`, `stale_base` and `submission_failed` describe other outcomes. |

Use the [record format](records/README.md) and the
[writing guide](../skills/precision-compare/references/community-records.md).
A draft accepts up to 256 KiB of supplied records and evidence. Structured evidence hashes
use UTF-8 JSON with two-space indentation, unescaped Unicode and a final newline,
in the supplied key order. The service validates additions and references against
local `origin/main`, binds the draft to that commit, and regenerates the search
index as part of the reviewed change. If upstream advances, fetch `origin/main`
and prepare a new draft. Edits also require a new preview and approval.
Maintainers review the PR; search sees new records after merge and deployment
of the new snapshot.

Configure a separate local stdio MCP server for contributions. Run from the
repository root, with the same dependency environment as search:

```sh
python -B -m community.mcp_submission --fork OWNER/NumericBridge
```

Use the authenticated owner's personal fork and GitHub CLI (`gh`). The client
must present MCP form elicitation to the user for a submission. A client that
does not support it can still prepare a draft. Configure this server only in
a client that obtains the user's answer instead of generating it. The public
HTTP endpoint remains read-only and does not require contributor credentials.

For clients that launch stdio servers from another directory, set the command
to the absolute Python executable with the MCP dependencies installed and set
`PYTHONPATH` to the repository root. Example for clients using `mcpServers`:

```json
{
  "mcpServers": {
    "numericbridge-community": {
      "command": "/path/to/venv/bin/python",
      "args": ["-B", "-m", "community.mcp_search", "--transport", "stdio"],
      "env": {"PYTHONPATH": "/path/to/NumericBridge"}
    },
    "numericbridge-contribution": {
      "command": "/path/to/venv/bin/python",
      "args": ["-B", "-m", "community.mcp_submission", "--fork", "OWNER/NumericBridge"],
      "env": {"PYTHONPATH": "/path/to/NumericBridge"}
    }
  }
}
```

The former `precision_community` package is now `community.records`. Existing
command flags remain the same. The former root `site/` moved to `community/site/`;
published URLs and paths relative to the static site remain unchanged. Historical
provenance strings inside frozen public files retain their original paths.
