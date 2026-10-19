"""Fixed-history inference capture and export (NumPy; Torch tensor for capture).

FixedHistory(sample, prompt, continuation).capture_and_force(logits, prefix,
rank=0, boundary='before_constraint') takes one full-vocabulary 1-D native Torch
logit row. It saves FP32 natural-log softmax before forcing the next fixed token
in place. Prefix.npy and prefix.json are new files; the returned JSON record
contains prediction position, complete history, target token, dtype and boundary.
Completed histories return None without modifying logits. The adapter selects
requests, gathers vocabulary shards and supplies logits before score processors.

validate_history(prompt, continuation, events) checks one prefill followed by
len(continuation)-1 decode events. Each event has stage=prefill|decode, input_ids,
positions (flat or repeated identical position rows), and prefill prefix_length.
The adapter maps native scheduling/chunks into these logical events.

export_records(records, base) returns {positions, scores, unavailable_targets}
for existing distribution/target-score manifests. Resolve scores.path relative
to base; exported paths are absolute. Distribution positions predict the target
at position+1. A zero-probability target has logprob=null and is listed as
unavailable for finite target-score metrics; its distribution retains -inf.
The caller supplies manifest settings, rank evidence and completion records.
"""
import json
import math
from pathlib import Path

import numpy as np


def _tokens(values):
    values = list(values)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('token IDs must be nonnegative integers')
    return values


class FixedHistory:
    def __init__(self, sample, prompt, continuation):
        if not isinstance(sample, str) or not sample:
            raise ValueError('sample must be nonempty')
        self.sample = sample
        self.prompt = _tokens(prompt)
        self.continuation = _tokens(continuation)
        if not self.prompt:
            raise ValueError('prompt must be nonempty')
        self.step = 0

    def capture_and_force(self, logits, prefix, rank=0, boundary='before_constraint'):
        if self.step == len(self.continuation):
            return None
        token = self.continuation[self.step]
        if logits.ndim != 1 or token >= logits.shape[0]:
            raise ValueError('expected a full vocabulary row containing target token')
        prefix = Path(prefix)
        array_path, meta_path = Path(str(prefix) + '.npy'), Path(str(prefix) + '.json')
        if array_path.exists() or meta_path.exists():
            raise FileExistsError(prefix)
        raw = logits.detach().float().log_softmax(dim=-1).cpu().numpy()
        if np.isnan(raw).any() or np.isposinf(raw).any() or raw.max() > 0:
            raise ValueError('invalid normalized logprobs')
        lp = float(raw[token])
        record = dict(sample=self.sample, step=self.step,
                      position=len(self.prompt)-1+self.step,
                      history=self.prompt+self.continuation[:self.step], token=token,
                      logprob=lp if math.isfinite(lp) else None,
                      logits_dtype=str(logits.dtype), logprob_dtype=str(raw.dtype),
                      rank=rank, boundary=boundary,
                      scores=dict(path=array_path.name, row=0))
        with array_path.open('xb') as stream:
            np.save(stream, raw, allow_pickle=False)
        with meta_path.open('x') as stream:
            json.dump(record, stream, allow_nan=False)
            stream.write('\n')
        logits.fill_(-float('inf'))
        logits[token] = 0
        self.step += 1
        return record


def validate_history(prompt, continuation, events):
    prompt, continuation = _tokens(prompt), _tokens(continuation)
    if not prompt or not continuation or len(events) != len(continuation):
        raise ValueError('expected one event per requested distribution')
    for step, event in enumerate(events):
        ids = prompt if step == 0 else [continuation[step-1]]
        positions = list(range(len(prompt))) if step == 0 else [len(prompt)+step-1]
        actual = event['positions']
        positions_match = actual == positions or (
            bool(actual) and all(row == positions for row in actual))
        if (event['stage'] != ('prefill' if step == 0 else 'decode')
                or event['input_ids'] != ids or not positions_match
                or (step == 0 and event.get('prefix_length') != 0)):
            raise ValueError(f'logical history mismatch at step {step}')
    return dict(distributions=len(events), prompt_tokens=len(prompt),
                decode_tokens=len(events)-1)


def export_records(records, base):
    base = Path(base).resolve()
    positions, scores, unavailable, seen = [], [], [], set()
    for record in records:
        sample, position, token = record['sample'], record['position'], record['token']
        coordinate = (sample, position)
        if coordinate in seen:
            raise ValueError(f'duplicate distribution coordinate: {coordinate}')
        seen.add(coordinate)
        if type(position) is not int or position < 0:
            raise ValueError('prediction position must be a nonnegative integer')
        history = _tokens(record['history'])
        if len(history) != position+1:
            raise ValueError('history must end at prediction position')
        _tokens([token])
        descriptor = dict(record['scores'])
        descriptor['path'] = str((base / descriptor['path']).resolve())
        positions.append(dict(sample=sample, position=position, history=history, scores=descriptor))
        target = dict(sample=sample, position=position+1, token=token)
        lp = record['logprob']
        if lp is None:
            unavailable.append(dict(**target, reason='zero target probability'))
        elif not math.isfinite(lp) or lp > 0:
            raise ValueError('target logprob must be finite and nonpositive or null')
        else:
            scores.append(dict(**target, logprob=lp))
    return dict(positions=positions, scores=scores, unavailable_targets=unavailable)
