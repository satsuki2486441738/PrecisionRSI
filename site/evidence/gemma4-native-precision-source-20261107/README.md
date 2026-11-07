# Gemma4 native precision source release

This is a **derived source archive**, not the exact executable used for the
recorded measurements and not a ready-to-launch cloud job. It contains the
allowlisted I01 cache-off, I03 partition/repeat, I04 padding/batch, GPU observer
and reserved48 input/collection/admission implementations, their CPU regressions,
and the relevant shared precision tools. Historical defect versions are retained
and labelled in the index. Reserved48 v3 identifies the frozen deployment source
candidate; including it does not assert that every planned native bin completed.

`source-release-v1.zip` uses the repository's `evidence_bundle.py` indexed ZIP
format. `EXPORT.json` records original/exported SHA256, byte counts and the
number of archive-level literal replacements. The sibling
`source-release-v1.index.json` additionally records each semantic transformation
rule and count. Its original-literal fields are hashes, not private literals.
A whole-file replacement is how the owning packer applies each derived source;
the detailed rule ledger explains the changes within that replacement.

Original source files and historical execution receipts remain unchanged.
Deployment paths and actual platform/device identities are replaced with inert
placeholders. Native launcher/collector entry points refuse before runtime
imports, and their module deployment gates are reset to `None`. Pure CPU
calculation, admission, input selection and replay interfaces keep their CLI and
callable interfaces. Original source pins identify original observed bytes;
they are **not** silently rewritten to authenticate sanitized code. Deployment
or admission in a different workspace requires explicit review and re-freezing
of that workspace's paths, versions, source hashes, allocation and finite claims.
No native execution or permission to spend an old claim is conveyed here.

The release process verifies the indexed ZIP, extracts it into a new directory,
parses all Python ASTs, checks native default refusals before framework imports,
and smoke-imports the safe CPU artifact helper and shared scalar/archive tools.
The original unmodified implementation tests and actual source-run evidence
provide the measurement provenance. These checks do not claim that every
sanitized regression can run without its documented runtime and evidence inputs.

## Public input provenance and licensing

The archive includes the **actual frozen public token inputs**: 48 reserved
article JSON files plus their manifest, eight discovery JSON files plus their
manifest, and the I04 prefix stress input derived from the first discovery
input. Prompts and forced continuations are token IDs from the public source,
not private user documents or model-generated quality labels. Their per-file
hashes are verified against the original manifests before release.

Source: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1` train
parquet SHA256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
The manifests retain original source row/article spans and text hashes for
attribution and deterministic regeneration. WikiText-derived data is supplied
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); preserve
attribution, these transformation records and share-alike obligations when
redistributing derived data. The repository source-code license remains separate.
The reserved set is English train-reserved material, not the official validation
split and not evidence of multilingual, code, mathematics or broad quality coverage.

Tokenizer/model identity is `google/gemma-4-12B` base, revision
`1a2c7729392586f5ecddae8807ee2ef741eabb9a`. The tokenizer SHA256 is
`42cacba08980fc39bfe2c73cb8c4486b9d00f2f2671072c6cb9c45ee2c5cfaae`.
No model weights, tokenizer binary, dataset parquet, runtime environment, wheel,
native/derived tensor arrays, private platform proof, credential, connection
profile, screenshot or raw execution log is bundled. Obtain dependencies and
model/data resources from their owners under the applicable licenses.

## Verification and use

From an existing checkout containing the shared precision tools:

```text
python skills/precision-compare/scripts/evidence_bundle.py verify source-release-v1.zip --max-expanded-bytes 67108864
python skills/precision-compare/scripts/evidence_bundle.py extract source-release-v1.zip NEW_EMPTY_DIRECTORY --max-expanded-bytes 67108864
```

Preserved repository-relative layout supports inspection of the collector,
schema, source-map, model-state/RNG, cache, BF16 encoding, CPU derivation and
comparison contracts together. Shared numerical tools remain the owners of the
original comparison math. Independent E02 verifiers use their documented
independent scalar arithmetic and diagnostic rounding tolerances; these are not
model acceptance thresholds. Public numerical milestone archives and target
replay scripts remain separate from this source-only release.

Actual original native/CPU outcome receipts are not recreated by extracting
this ZIP. Full task completion, numerical acceptance, billed cost and an
independent full native replay are not inferred from source publication.

## Upstream license notices

The [pinned WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) describes CC BY-SA 4.0 in its prose; its metadata also lists CC BY-SA 3.0 and GFDL. The archive and index retain the experimental manifests and the card's stated 4.0 label. Preserve upstream attribution and license notices with the derived inputs.

## Published results

- [H100 and Ascend cache-off anchor](https://satsuki2486441738.github.io/PrecisionRSI/#evaluation-gemma4-h100-ascend-cacheoff-20261106)
- [H100 and Ascend partition and repeat check](https://satsuki2486441738.github.io/PrecisionRSI/#evaluation-gemma4-h100-ascend-partitions-20261107)
- [H100 and Ascend batch and padding check](https://satsuki2486441738.github.io/PrecisionRSI/#evaluation-gemma4-h100-ascend-batch-padding-20261107)
- [H100 observer transparency](https://satsuki2486441738.github.io/PrecisionRSI/#evaluation-gemma4-h100-observer-transparency-20261107)
