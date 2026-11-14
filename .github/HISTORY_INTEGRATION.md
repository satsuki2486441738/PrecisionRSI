# Evaluation integration planning

`plan_history_integration.py` builds a private append-only candidate from a base
evaluation history and named source histories. It does not publish records,
modify inputs, copy evidence, merge branches, or establish numerical acceptance.
Use it before moving privately retained measurements into the public catalog.

```sh
python -B skills/precision-compare/scripts/record_run.py \
  --output-root runs --side compare --task history-integration --timeout 600 \
  -- python -B .github/plan_history_integration.py \
  --base site/data/evaluations.json \
  --source supplement=/private/staging/site/data/evaluations.json \
  --output /private/new-integration-plan
```

Use a fresh private output directory. Keep generated plans out of commits; they
contain input paths and may contain private records. Each input's parent `site`
directory must contain its evidence paths. For a Git branch, stage its catalog
and relevant evidence together into a private directory first. The tool never
fetches remote URLs or treats a local Git revision as publicly reachable.

The base order and metadata are retained. New IDs append in source order;
identical records deduplicate while retaining every source membership. Different
content under the same ID produces an explicit conflict, exit code 2, and no
candidate. Missing IDs, duplicate IDs/JSON keys, incompatible schemas and
non-finite JSON values reject the input. Dates, coverage, status, acceptance,
settings and metrics remain unchanged. A candidate is not a published snapshot;
its base `updated_at` deliberately remains unchanged.

`audit.json` pins input SHA-256 values and inspects direct evidence references
for additions. It reports missing or escaping paths, unresolved external URLs,
explicit private archive provenance, unverified archive revision reachability,
files over 50 MiB, unreviewed archives and selected runtime identifiers. Text
scanning is bounded to 8 MiB per file. Archives are hashed but never unpacked.
This check is deliberately incomplete: a clean scan does not certify privacy,
replayability, scientific validity or publication readiness. Every addition
requires review, including evidence shared with another input.

Publish newly derived statistics and sanitized reports under new paths when raw
artifacts contain private execution details. Describe retained private evidence,
replay limits and provenance accurately. Preserve original records and evidence;
do not silently replace historical archive links or rewrite partial runs as
completed. Public hardware aliases can be supplied in a separate presentation
mapping without changing original source bytes. Compare observations do not
establish a confirmed Repair or hardware causality. Preserve unknown acceptance.

Before proposing a catalog update, run the history, differences and immutability
validators, the full public unit suite and repository validation. Frontend
integration is a separate task; this planner changes no page or UI contract.
