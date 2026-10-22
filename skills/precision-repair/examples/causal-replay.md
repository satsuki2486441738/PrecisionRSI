# Trace a training discrepancy

Illustrative task: repair an update discrepancy for scalar SGD with learning rate
`0.1`, initial parameter `w=0`, and targets `[1, 3]` split into two microbatches.
The intended objective is `L(w) = mean((w-y)**2)/2`; its analytic gradient is `w-2`.

1. **Retain the failing workload.** Record the native entrypoint and selected
   implementation, source/runtime identities, inputs and initial state. Capture
   each step's parameter before the update, accumulated gradient and stored
   parameter afterward.
2. **Compare at matching states.** Evaluate the analytic gradient at each captured
   native parameter. Separately advance an oracle trajectory from the original
   initial state using the intended gradient and learning rate.

   | Step | Native parameter before | Native gradient | Expected gradient at native state | Oracle parameter before | Oracle gradient |
   |---|---:|---:|---:|---:|---:|
   | 1 | 0 | -4 | -2 | 0 | -2 |
   | 2 | 0.4 | -3.2 | -1.6 | 0.2 | -1.8 |

   The local gradient is twice its expected value at both native states. The
   second-step trajectory difference also includes the changed parameter state;
   comparing only `-3.2` with `-1.8` obscures that local scaling error.
3. **Test the cause.** Inspect how microbatch losses are reduced over the optimizer
   window. Check the hypothesis that microbatch means were summed without window
   normalization; use a full-batch execution as a control for the same objective.
4. **Patch and verify.** Add a regression against the analytic objective, confirm
   it fails on the original path, then fix the responsible reduction. Rerun from
   `w=0`: expected gradients are `[-2, -1.8]` and stored parameters `[0.2, 0.38]`.
   Verify both steps and nearby batch partitions with a justified dtype tolerance.
5. **Report both views.** Retain same-state local errors and original-trajectory
   before/after errors, paired-coordinate counts, patch and regression results.
   Compare actual stored updates using the existing tensor tools.
