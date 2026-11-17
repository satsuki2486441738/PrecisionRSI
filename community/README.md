# NumericBridge community

This directory owns the community website and its public record tools together.

| Directory | Purpose |
|---|---|
| [site/](site/README.md) | Static pages, public JSON records and selected evidence |
| [records/](records/README.md) | Python validation, read-only search and explicit relations |

```sh
python -B -m community.records validate
python -B -m community.records index --output /tmp/numericbridge-records.json
python -m http.server 8000 --directory community/site
```

Run commands from the repository root. Publish only `community/site/`; Python
code and maintenance documentation are not included in the Pages deployment.
The website reads JSON directly. Record tools run locally or in CI; there is no
online Python API, account service or database. They do not run model workloads.

The former `precision_community` package is now `community.records`. Existing
command flags remain the same. The former root `site/` moved to `community/site/`;
published URLs and paths relative to the static site remain unchanged. Historical
provenance strings inside frozen public files retain their original paths.
