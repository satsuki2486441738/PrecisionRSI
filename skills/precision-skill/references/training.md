# SFT and RL comparison

## Same-state SFT step

Start both sides from the same parameters and optimizer state, with a trustworthy parameter/shard mapping. Freeze optimizer formula and hyperparameters, step, scheduler, master weights, loss scaling, batch/microstep order, accumulation, global token normalization, communication, and clipping stage.

Record the scalar objective and its definition, gradients at the agreed stage, and actual parameter changes. A completed `optimizer.step()` is not evidence of a nonzero update: save the required before/after values and compute their differences. Use the shared coordinate rules and relative-L2 definition in [comparison](comparison.md).

Record effective train/eval mode, dropout, sampling, and random operators. Matching seeds alone do not establish identical random quantities across stacks. Supply common random quantities or a controlled input when necessary; distinguish a controlled variant from the native path and state the remaining limits.

Freeze selection before collection when observing sampled parameters or shards. Save tensor identity, logical shape, global coordinates, counts, and shard mapping. Pair the complete agreed sampled set rather than silently keeping only its intersection. Conclusions apply to the saved coordinates; sparse observations do not establish full-gradient or full-update agreement.

A complete training-state comparison requires evidence of actual restoration of that state. Weight restoration or a small tensor readback establishes only its own scope. Save environment association at launch using [environment](environment.md), so later comparison can reuse either the runtime inventory or associated locked-environment records.

## RL dependencies

Choose the actual RL objective; do not infer PPO from the task label. Freeze prompt/response trajectories, valid tokens, termination/truncation rules, and the objective's old/reference logprobs, rewards, advantages, returns, and their provenance.

Check clipping, KL definition and weight, and token/sequence normalization. Common injected dependencies establish downstream comparability; their upstream production needs separate coverage. Distinguish fixed-trajectory numerical validation from online rollout behavior or task-quality claims.

## State and trajectory divergence

| Mode | Required interpretation |
|---|---|
| Same-state numerical comparison | The observation point has common inputs, initial state, objective dependencies, and actual random-condition evidence. |
| Free trajectory or training effectiveness | States/history may evolve; separately freeze task, dataset split, generation settings, and outcome metrics. |
| Same-device version regression | Fix the device and other conditions; identify the reference version and intended change. |

After the first update, parameters and optimizer states can diverge. Later steps compare trajectories unless a common state is restored for each same-state step. After generation/rollout divergence, retain both histories and stop interpreting different tokens at equal indices as common-input observations.

Numerical consistency and task effectiveness answer separate questions. Scoring evidence alone does not establish backward or update execution.

## Collector and comparison readiness

Before device execution, specify the objective, input replay, effective modes, normalization, gradient stage, optimizer initial state, coordinate mapping, saved observations, completion checks, and environment association. Accept a minimal collection before expanding it.

The bundled schema-v2 score comparator implements full-forward logprobs. Prepare case-level training collectors and validators for losses, gradients, and actual updates; preserve their format/version and evidence with the case. Report actual coverage of objective, backward, and update separately. Use a frozen threshold when acceptance is requested; otherwise report numerical diagnostics.
