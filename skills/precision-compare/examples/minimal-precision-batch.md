# Fixed precision batch

Load the bundled [100-example v1 dataset](precision-batch-v1.json) directly from
`examples/precision-batch-v1.json` relative to the `precision-compare` Skill root.
In the repository, it lives at `skills/precision-compare/examples/precision-batch-v1.json`.
It contains coin, chip draw, spinner and die events, each with a unique visible
ID and a fixed result.
Inference and SFT prompts state the result; RL prompts ask for a guess and leave
the result in the separate answer key.

The [v1 baseline lock](precision-batch-v1.lock.json) pins the exact 100-event
batch. When no dataset is specified, use all 100 events in the same order on
both sides of the formal comparison. Preflight can use the smallest valid subset;
diagnostic replay can use the events that triggered the difference. Event count
does not set the number of samples per event, groups per update or updates.
The installation smoke check verifies the bundled copy against the lock and the
output of the [generator](generate_precision_batch.py). To reproduce it, run
from the `precision-compare` Skill root:

```sh
python -B examples/generate_precision_batch.py --count 100 --seed 17 \
  --group-size 4 --output NEW.json
```

For a comparison, record the batch ID and hash, the selected event IDs in
execution order, and the chosen views and groups. Preserve v1 when changing the
fixture; publish a new dataset version for any change to its generated content.

The dataset contains `inference`, `sft`, `rl_online`, `rl_replay` and
`answer_key` views linked by
`id`. Join `rl_replay` with `rl_online` for its prompt. Use only the relevant
`prompt` as model input, and keep `answer_key` with the evaluator. The original
[four-prompt fixture](minimal-precision-batch.json) is available for a manual
smoke check. Freeze rendered token IDs, masks and initial states once the model
and runtime are selected.

| Path | Use the batch to check |
|---|---|
| Inference | Score the fixed A/B continuations at matching histories. Use short and long prompts together when checking batch padding. If the claimed path includes decode after the first output, extend the fixed continuation and verify the required native decode positions. |
| SFT | Use the visible result and its `completion`. Compare the effective loss mask, loss, gradients and one optimizer update through the selected training path. |
| RL, fixed replay | Use the fixed `completions`; the included `rewards` are expected scoring results. After stripping surrounding whitespace, reward 1 for an exact target match and 0 otherwise. Keep actual rewards and the effective action mask. Check a mixed group for nonzero reward-driven reference gradients and use the all-correct and all-wrong groups as separate controls for the group-centered reward term. |
| RL, online | Sample from the selected policy with the same prompts and reward rule. Record actual groups and invalid completions separately. Claim mixed reward coverage only for valid actions that reached the objective. Verify the next rollout uses the updated policy when weight handoff is in scope. |

For the mixed replay, check the reward-driven gradient before aggregating
different prompts: opposing groups can cancel. Fixed choices are replay inputs;
retain the native old-policy scores and report which dependencies were injected
or recomputed, as in the [RL recipe](../references/rl.md#collection).

Extend this batch only to exercise an uncovered path, such as a minimum group,
accumulation window, collective, or context length. Record which condition
triggered and which observations were paired. The batch measures numerical
behavior, not task accuracy or learning quality.
