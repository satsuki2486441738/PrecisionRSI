# Community submission guide

After delivery, use this guide to write a public record if the user opts in to
sharing. For the submission procedure, see [the workflow](../../numericbridge/references/workflow.md#resume-and-deliver).

## Record the result

| Work completed | What to record |
|---|---|
| Compared two configurations | The tested configurations, workload, coverage and measured differences. A comparison can stand on its own. Record acceptance separately when criteria were supplied. |
| Investigated a difference | Link the comparison to the investigation. Describe the observed issue and the evidence for its classification. |
| Evaluated a solution | Link each patch, upgrade or settings change to the issue it addresses. Give each evaluated solution its own verification, with the tested configuration, workload, measured outcome and tradeoffs. Keep a source-only finding distinct from an executed check. |

To make the submitted record searchable, record the hardware, model checkpoint
and revision, tested framework and version, and actual task and settings. Use
`model.architecture` and `model.size` for search facets. Use `tiny` for a custom
miniature fixture; retain its dimensions and seed with the experiment. List
tested frameworks in `frameworks`, dependencies in `libraries`, and the repair
target in `affected_library`. Keep release versions and commits in their fields;
put patch or backport descriptions on the solution.

## Write the public entry

Use a short title for the task or symptom, for example "LoRA output changes
after merging". Use the summary for the observed effect, the change or a
condition that affects use. Put versions, settings and measurements in their
structured fields. Keep implementation detail with the technical evidence.

Use `content.title` and optional `content.summary` in `zh` and/or `en`;
lifecycle entities use `details.content`. `content.workflows` lists affected
workflows; the experiment configuration records what actually ran. To clarify
historical wording, use the typed ID in `data/content.json`.

Include the minimum public evidence needed to trace each measurement. Keep raw
logs, prompts, private paths and host details with the local experiment. Retain
public hardware, model and software identities needed to interpret the result.
