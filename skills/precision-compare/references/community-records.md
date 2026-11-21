# Community submission guide

Select public findings from the local issue record and linked evidence. Follow
[the workflow](../../numericbridge/references/workflow.md#community-contribution)
for preview and consent.

## Record the result

| Work completed | What to record |
|---|---|
| Comparison | Tested configurations, workload, coverage and measured differences. It can stand alone; record acceptance separately. |
| Diagnosis | Link the observed issue and investigation to the comparison or native failure. Cite the evidence for its classification. |
| Solution | Link each candidate to the issue and record its own scope, configuration, outcome and tradeoffs. Distinguish source review, workaround and original-path verification. |

Record hardware, checkpoint and revision, tested framework/version, task and
settings. Use `model.architecture` and `model.size` for search; use `tiny` for a
custom miniature fixture and retain its dimensions and seed. Use `frameworks`
for tested frameworks, `libraries` for dependencies and `affected_library` for
the repair target. Keep versions and commits in their own fields.

## Write the public entry

Describe the symptom or observed effect in a short title and summary; keep
settings and metrics in structured fields. Use `content.title` and optional
`content.summary` in `zh` and/or `en` (`details.content` for lifecycle entities).
`content.workflows` describes affected tasks; experiment settings describe what
ran. Clarify historical wording by typed ID in `data/content.json`.

Include only the evidence needed to trace the result. Keep raw logs, prompts,
private paths and host details local; retain public stack identities.

## Prepare an upstream PR draft

For a verified patch intended for upstream, identify the repository and base
branch. Read that branch's contribution guide, PR template and relevant linked
requirements. Use the template, or **Problem / Change / Validation / Patch**
if absent or unreadable; record which guidance could not be checked.

Include a title, any existing issue link, affected setup, minimal reproducer,
matched before/after results, commands run and the patch or public link. Keep
pending checks and attestations unchecked. Record a PR URL only once it exists.

Attach the draft through the record's `evidence` reference. In the
`prepare_community_submission` evidence map, add `upstream_pr_draft` to a new
`evidence/submissions/*.json` document with `repository`, `base_branch`, `title`,
`body_markdown`, `guidance_urls` and `pending_checks`. Put the patch or link in
the body or same document; the regular preview and consent cover it.

## Local preview

If contribution MCP is unavailable, retain the anonymized input. From the
repository checkout, validate the local draft:

```sh
python -B -m community.records.submission --input submission.json --output preview.json
```
