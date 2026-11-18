# Token-mean evidence scope clarification

Record: `transformers-530-qwen35-token-mean-20261024`.

This additive clarification preserves the original report and numerical evidence.
The historical Markdown and the JSON case classification use stronger defect
wording than the retained workload establishes. The catalog remains investigating.

The executed scope is manual native `Trainer.training_step`, selected gradients
and a shared Accelerator accumulation scale. It does not include native
`Trainer.train`, an optimizer update, independent absolute-gradient scaling,
long training or multi-device validation. The retained before/after comparisons
support a token-weighted objective candidate under that scope; they do not yet
establish an upstream library defect or numerical acceptance.

The new issue is therefore an unconfirmed hypothesis and its numerical-policy
solution has partial retained-measurement verification. No historical measurement,
acceptance field, patch, or report is rewritten. Native full-loop semantics and
independent scaling certification remain unresolved.

Sources: [retained report](transformers-530-qwen35-token-mean-20261024.md),
[retained numerical evidence](transformers-530-qwen35-token-mean-20261024.json),
and the unchanged catalog's scope, upstream status and remaining limitations.
