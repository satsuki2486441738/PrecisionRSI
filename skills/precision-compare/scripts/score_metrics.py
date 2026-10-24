"""Framework-independent natural-log target-score arithmetic; standard library.
Call: reduce_scores(side_a, side_b)
Input: Matching coordinate-tuple maps to finite natural logprobs <= 0.
Output: Dict keys target_count, side_a_nll_nat_per_token, side_b_nll_nat_per_token,
nll_abs_diff_nat_per_token, mae_nat, p99_nat, p99_method, max_abs_nat, max_coordinate.
Requires: Standard library; equal nonempty inventories; task adapters own masks, shifts and objectives.

Details:
reduce_scores(side_a, side_b) accepts mappings from hashable coordinate tuples
(string/integer components) to finite logprobs <= 0. Adapters own masks, token
identities, native objectives and grouping. Coordinate sets must match exactly.
Returns target-weighted NLL per side, absolute NLL difference, MAE, linear P99,
maximum absolute error and its coordinate. Ties use canonical coordinate order.
No task-specific label shift, EOS, dialogue or cache policy is imposed here.
"""
import json
import math


def reduce_scores(side_a, side_b):
    if not side_a or set(side_a) != set(side_b):
        raise ValueError('nonempty equal target coordinate sets required')
    for key in side_a:
        if not isinstance(key, tuple) or not key or any(type(x) not in (str, int) for x in key):
            raise ValueError('coordinates must be nonempty tuples of strings/integers')
    keys = sorted(side_a, key=lambda k: json.dumps(k, ensure_ascii=False, separators=(',', ':')))
    for values in (side_a, side_b):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values.values()):
            raise ValueError('finite natural logprobs <= 0 required')
    errors = [abs(side_a[k] - side_b[k]) for k in keys]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError('nonfinite target-score difference')
    count = len(keys)
    nll = [math.fsum(-values[k] for k in keys) / count for values in (side_a, side_b)]
    worst = max(range(count), key=errors.__getitem__)
    ordered = sorted(errors)
    h = 0.99 * (count - 1)
    lo, hi = math.floor(h), math.ceil(h)
    result = dict(target_count=count, side_a_nll_nat_per_token=nll[0],
                  side_b_nll_nat_per_token=nll[1], nll_abs_diff_nat_per_token=abs(nll[0]-nll[1]),
                  mae_nat=math.fsum(errors)/count,
                  p99_nat=ordered[lo]+(h-lo)*(ordered[hi]-ordered[lo]),
                  p99_method='global linear h=0.99*(N-1)', max_abs_nat=errors[worst],
                  max_coordinate=list(keys[worst]))
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError('nonfinite score metric')
    return result
