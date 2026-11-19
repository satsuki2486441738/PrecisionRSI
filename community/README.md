# NumericBridge community

This directory owns the community website and its public record tools together.

| Directory | Purpose |
|---|---|
| [site/](site/README.md) | Static pages, public JSON records and selected evidence |
| [records/](records/README.md) | Python validation, read-only search and explicit relations |
| [mcp_search.py](mcp_search.py) | Optional MCP server for Agent access to community search |
| [mcp_submission.py](mcp_submission.py) | Local previews and confirmed uploads |
| [intake.py](intake.py) | Private community inbox for received submissions |

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
prepares an anonymized preview and asks once before sending it to the community.
The contributor needs no GitHub account or fork.

| Operation | Input | Result |
|---|---|---|
| `prepare_community_submission` | `title` plus optional arrays `evaluations`, `repairs`, `library_groups`, `entities` and `links`, and an `evidence` map of new `evidence/submissions/*.json` paths to JSON objects. | `ready` returns `draft_id`, `draft_sha256` and the exact local `preview`; `invalid` returns findings. This does not upload data. |
| `submit_community` | `draft_id` and `draft_sha256` after showing the preview. The MCP form asks for consent. | `received` returns `receipt_id`, `review_state: pending_review`, and `reused`. The same upload returns the original receipt. Other outcomes include `declined`, `cancelled`, `approval_unavailable`, `intake_unavailable`, `invalid_draft` and `submission_failed`. |

Use the [record format](records/README.md) and
[writing guide](../skills/precision-compare/references/community-records.md).
A draft accepts up to 256 KiB of records and minimum evidence. Evidence hashes
use UTF-8 JSON, two-space indentation, unescaped Unicode and a final newline in
the supplied key order. Preview validation uses local `origin/main`; the inbox
also validates additions against the community catalog before storing them.
Changing the content or inbox destination requires a new preview and consent.

Configure a stdio MCP server with the community's intake URL and an upload
credential issued by its operator. When enabled, the published site advertises
the API address in `data/community-service.json`; `submission: null` means no
endpoint is configured. The following uses a placeholder hostname:

```sh
python -B -m community.mcp_submission \
  --intake https://community.example.org/v1/submissions --token-file /path/to/upload-token
```

The client must present the MCP form to the user. In clients that launch from
another directory, use the absolute Python executable and set `PYTHONPATH` to
the NumericBridge checkout. Omit both options to prepare local drafts only. If
MCP is unavailable, the equivalent preview command is:

```sh
python -B -m community.records.submission --input submission.json --output preview.json
```

The input contains `title` and `additions`. The `additions` groups are
`evaluations` (`evaluations`), `repairs` (`repairs`, `library_groups`), `relations`
(`entities`, `links`) and `evidence` (path to JSON object). Use empty objects for
unused groups. The command writes a new private `local_draft` file; reuse its
additions once an inbox is configured.

### Community operator setup

GitHub Pages hosts the catalog and endpoint descriptor. Run the write API on a
community host with HTTPS and persistent private storage. For a Linux host:

1. Create a dedicated `numericbridge` service user and group. Check out the
   reviewed source at `/opt/numericbridge` with a current `origin/main` ref.
   Own the checkout as the service user so Git can read it, then install dependencies:

   ```sh
   python3 -m venv /opt/numericbridge/.venv
   /opt/numericbridge/.venv/bin/python -m pip install -r /opt/numericbridge/community/requirements-mcp.txt
   ```

2. Put a generated URL-safe upload token in `/etc/numericbridge/upload-token`
   (root-owned, mode `0600`). Install the
   [systemd unit](deploy/numericbridge-intake.service) and run
   `sudo systemctl enable --now numericbridge-intake`. The service binds to
   loopback; save and back up `/var/lib/numericbridge/inbox` separately from the
   code. Restart the service when rotating the upload token.
3. Point a community subdomain at the host. With Caddy 2.10 or later, use the
   [reverse proxy config](deploy/Caddyfile) and set `NUMERICBRIDGE_INTAKE_HOST`
   to that hostname in the Caddy service environment. The proxy routes only
   `POST /v1/submissions` and `GET /healthz`. Set an appropriate request rate
   limit at the public ingress. An existing HTTPS proxy can use the same routes.
4. Call `GET /healthz` over HTTPS, check that an unauthorized `POST` returns
   `401`, then test with a synthetic draft. Set the GitHub Actions repository
   variable `NUMERICBRIDGE_INTAKE_URL` to the verified public HTTPS endpoint and
   run the Pages workflow. Distribute the upload credential privately; only
   the endpoint URL belongs in the Pages build.

For local testing, run `python -B -m community.intake --inbox PATH --token-file PATH`
and use `http://127.0.0.1:8770/v1/submissions` as the MCP intake URL.
Keep `origin/main` current on the review host. Each `submission-<sha256>.json`
holds the approved bytes for maintainers to review. Maintainers decide what to
add to the catalog and whether to contact upstream. A case is searchable once
the deployed index returns its record ID.

The former `precision_community` package is now `community.records`. Existing
command flags remain the same. The former root `site/` moved to `community/site/`;
published URLs and paths relative to the static site remain unchanged. Historical
provenance strings inside frozen public files retain their original paths.
