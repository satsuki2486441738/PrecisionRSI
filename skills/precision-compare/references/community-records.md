# Community record writing

Use this guide when preparing a requested community submission or updating public
descriptions. Publishing follows the user's authorized destination and scope.

Write the observation and proposed change in plain language:

- Title: the problem or task, such as “LoRA output changes after merging”.
- Summary: the observed effect or the solution's change, in one or two sentences.
- Configuration: hardware, model architecture, size, checkpoint and revision,
  tested frameworks, library versions and settings. Use `model.architecture` and `model.size` for
  the search facets; use `tiny` for a custom miniature fixture and keep its
  actual dimensions and seed in the original record.
- Verification: workload, coverage, measured values, outcome and evidence.
- Tradeoffs or remaining work: the specific condition that affects use, if any.

Keep public wording in `content.title` and optional `content.summary`, localized
with `zh` and/or `en`. Lifecycle entities use `details.content`. Historical
descriptions can be maintained by typed ID in `data/content.json`. Optional
`content.workflows` lists affected inference/SFT/RL usage; actual executed tasks
remain part of the experiment configuration.

Use structured fields for versions, commits, settings, status and relationships.
List the frameworks evaluated on each side in `frameworks`; keep the complete
dependency stack in `libraries`. Use `affected_library` for the repair target.
Keep `libraries[].version` as the version identifier. Describe a local patch,
source backport or upgrade in its solution record rather than in a version string.
Keep code paths, implementation steps and detailed reproduction material with
the technical record. A source-inspected upgrade and an executed repair have
different verification records; express that through their status and evidence.

Omit repeated stock caveats such as “limited to this verification” or “not a
global pass”. Record concrete conditions where they matter, such as the tested
adapter type or an outstanding workload. Omit optional fields when there is
nothing useful to add.

Link each solution to the issue it addresses and each verification to its
solution and workload. Multiple patches, upgrades and numerical policies can
coexist, with their own measurements and tradeoffs.

Search results supply context for the requested task. No match means there is
no community context to reuse; continue the normal comparison/repair workflow.
Search by the framework the user runs or the library they need to repair.
Use a model group to browse results, then select an exact checkpoint for
recommendations. Present the tested checkpoint and briefly explain any missing
requirement when the user requests a recommendation.
