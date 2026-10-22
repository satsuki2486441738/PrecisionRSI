# Trace a training discrepancy

Consider scalar SGD with learning rate `0.1`, initial parameter `w=0`, and targets
`[1, 3]` split into two microbatches. The intended objective is
`L(w) = mean((w-y)**2)/2`; its analytic gradient is `w-2`.

| Step | Native parameter before | Native gradient | Expected gradient at native state | Correct trajectory parameter before | Correct trajectory gradient |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | -4 | -2 | 0 | -2 |
| 2 | 0.4 | -3.2 | -1.6 | 0.2 | -1.8 |

The local gradient is twice its expected value at both native states. At step 2,
comparing only `-3.2` with `-1.8` also includes the changed parameter state.
Same-state replay isolates the reduction error: the implementation summed the
microbatch means without window normalization. A full-batch execution of the
same objective provides a control.

A regression against the analytic objective distinguishes the original reduction
from the fix. Rerunning from `w=0` then gives gradients `[-2, -1.8]` and stored
parameters `[0.2, 0.38]`. Together these checks establish the local correction
and recovery of the two-step workload; the same execution can supply both results.
