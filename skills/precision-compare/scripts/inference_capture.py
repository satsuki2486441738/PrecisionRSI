"""Fixed-history inference capture and export (NumPy; Torch tensor for capture).

FixedHistory(sample, prompt, continuation, logprob_dtype='float32').capture_and_force(logits, prefix,
rank=0, boundary='before_constraint') takes one full-vocabulary 1-D native Torch
logit row. It saves FP32 natural-log softmax before forcing the next fixed token
in place. Prefix.npy and prefix.json are new files; the returned JSON record
contains prediction position, complete history, target token, dtype and boundary.
The default retains native-device Torch FP32 log_softmax. Explicit float64
copies original values to CPU and derives NumPy FP64 logprobs offline; this does
not change native model/cache arithmetic or claim native FP64 probabilities.
Completed histories return None without modifying logits. The adapter selects
requests, gathers vocabulary shards and supplies logits before score processors.

validate_history(prompt, continuation, events) checks one prefill followed by
len(continuation)-1 decode events. Each event has stage=prefill|decode, input_ids,
positions (flat or repeated identical position rows), and prefill prefix_length.
Token/position/prefix values must be integers, not booleans or floats. Decode
prefix_length, when supplied, must equal the preceding causal history length.
The adapter maps native scheduling/chunks into these logical events.

derive_logprobs(logits, dtype='float64') accepts floating NumPy [vocabulary] or
[positions,vocabulary] raw logits and returns newly derived stable natural-log
softmax in the explicitly selected float32/float64 dtype. Record the derivation
separately from native score dtype; original raw rows are never modified.

validate_generation_scores(context_logits, generation_logits, decode_logits=None)
checks unmodified NumPy [positions,vocabulary] native rows: first generation
must equal the LAST context row, and supplied cached decode rows must equal all
remaining generation rows. It checks exact identity, not numerical precision
between engines. It cannot establish physical positions or cache dtype; pair
with validate_history and the adapter's actual native execution observations.

export_records(records, base, normalization_tolerance=1e-5, score_tolerance=0.0)
returns {positions, scores, unavailable_targets}
for existing distribution/target-score manifests. Resolve scores.path relative
to base; exported paths are absolute. Distribution positions predict the target
at position+1. A zero-probability target has logprob=null and is listed as
unavailable for finite target-score metrics; its distribution retains -inf.
Before exporting, load each selected .npy/.safetensors row, check full-vocabulary
normalization, legal tokens, row selection and agreement of the supplied target
logprob with that exact row. None is valid only for an actual -inf target.
Tolerances are explicit nonnegative absolute diagnostics, not precision acceptance
thresholds. The caller supplies manifest settings, ranks and completion records.
"""
import json
import math
from pathlib import Path

import numpy as np
from artifact_io import Array


def derive_logprobs(logits, dtype='float64'):
    """Derive offline logprobs without editing native scores or relaxing mass checks."""
    if dtype not in ('float32', 'float64'):
        raise ValueError('logprob dtype must be float32 or float64')
    values = np.asarray(logits)
    if (values.ndim not in (1, 2) or not values.size
            or values.dtype.kind != 'f' or values.dtype.itemsize not in (2, 4, 8)
            or np.isnan(values).any() or np.isposinf(values).any()
            or not np.isfinite(values).any(axis=-1).all()):
        raise ValueError('expected nonempty floating logit rows with finite support')
    support = np.isfinite(values)
    with np.errstate(over='ignore'):
        values = values.astype(dtype, copy=True)
    if not np.array_equal(support, np.isfinite(values)):
        raise ValueError('logits cannot be represented in the selected derivation dtype')
    with np.errstate(over='ignore', under='ignore'):
        values -= values.max(axis=-1, keepdims=True)
        values -= np.log(np.exp(values).sum(axis=-1, keepdims=True, dtype=dtype))
    if not np.array_equal(support, np.isfinite(values)):
        raise ValueError('logprobs cannot be represented in the selected derivation dtype')
    return values


def validate_generation_scores(context_logits, generation_logits, decode_logits=None):
    """Bind generation score coordinates to their authoritative native boundaries."""
    context, generation = np.asarray(context_logits), np.asarray(generation_logits)
    for values in (context, generation):
        if (values.ndim != 2 or not values.size or values.dtype.kind != 'f'
                or np.isnan(values).any() or np.isposinf(values).any()
                or not np.isfinite(values).any(axis=-1).all()):
            raise ValueError('expected nonempty floating context/generation score matrices')
    if context.shape[1] != generation.shape[1]:
        raise ValueError('context/generation vocabulary mismatch')
    if not np.array_equal(context[-1], generation[0]):
        raise ValueError('first generation must equal the last context score row')
    if decode_logits is not None:
        decode = np.asarray(decode_logits)
        if (decode.shape != generation[1:].shape or decode.dtype.kind != 'f'
                or not np.array_equal(decode, generation[1:])):
            raise ValueError('cached decode score rows do not match generation rows')
    return dict(distributions=len(generation), prompt_rows=len(context),
                decode_rows=len(generation)-1, first_generation_is_last_context=True,
                decode_rows_checked=decode_logits is not None)


def _tokens(values):
    values = list(values)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('token IDs must be nonnegative integers')
    return values


class FixedHistory:
    def __init__(self, sample, prompt, continuation, logprob_dtype='float32'):
        if not isinstance(sample, str) or not sample.strip():
            raise ValueError('sample must be nonempty')
        self.sample = sample
        self.prompt = _tokens(prompt)
        self.continuation = _tokens(continuation)
        if not self.prompt:
            raise ValueError('prompt must be nonempty')
        if logprob_dtype not in ('float32', 'float64'):
            raise ValueError('logprob dtype must be float32 or float64')
        self.logprob_dtype = logprob_dtype
        self.step = 0

    def capture_and_force(self, logits, prefix, rank=0, boundary='before_constraint'):
        if self.step == len(self.continuation):
            return None
        if type(rank) is not int or rank < 0:
            raise ValueError('rank must be a nonnegative integer')
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError('boundary must be nonempty')
        token = self.continuation[self.step]
        if not logits.is_floating_point() or logits.ndim != 1 or token >= logits.shape[0]:
            raise ValueError('expected a full vocabulary row containing target token')
        prefix = Path(prefix)
        array_path, meta_path = Path(str(prefix) + '.npy'), Path(str(prefix) + '.json')
        if array_path.exists() or meta_path.exists():
            raise FileExistsError(prefix)
        if self.logprob_dtype == 'float32':
            raw = logits.detach().float().log_softmax(dim=-1).cpu().numpy()
            normalization = 'Torch log_softmax on float32 native-device logits'
        else:
            raw = derive_logprobs(logits.detach().cpu().double().numpy())
            normalization = 'Offline NumPy float64 log_softmax of copied native values'
        if np.isnan(raw).any() or np.isposinf(raw).any() or raw.max() > 0:
            raise ValueError('invalid normalized logprobs')
        lp = float(raw[token])
        record = dict(sample=self.sample, step=self.step,
                      position=len(self.prompt)-1+self.step,
                      history=self.prompt+self.continuation[:self.step], token=token,
                      logprob=lp if math.isfinite(lp) else None,
                      logits_dtype=str(logits.dtype), logprob_dtype=str(raw.dtype),
                      normalization=normalization,
                      rank=rank, boundary=boundary,
                      scores=dict(path=array_path.name, row=0))
        with array_path.open('xb') as stream:
            np.save(stream, raw, allow_pickle=False)
        with meta_path.open('x', encoding='utf-8') as stream:
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
        if not isinstance(event, dict):
            raise ValueError(f'logical history event must be an object at step {step}')
        ids = prompt if step == 0 else [continuation[step-1]]
        positions = list(range(len(prompt))) if step == 0 else [len(prompt)+step-1]
        actual = event.get('positions')
        if not isinstance(actual, list) or not actual:
            raise ValueError(f'invalid position IDs at step {step}')
        rows = actual if isinstance(actual[0], list) else [actual]
        positions_match = all(isinstance(row, list) and _tokens(row) == positions for row in rows)
        prefix = event.get('prefix_length')
        prefix_expected = 0 if step == 0 else len(prompt)+step-1
        prefix_match = ((step > 0 and 'prefix_length' not in event)
                        or (type(prefix) is int and prefix == prefix_expected))
        if (event.get('stage') != ('prefill' if step == 0 else 'decode')
                or _tokens(event.get('input_ids', [])) != ids or not positions_match
                or not prefix_match):
            raise ValueError(f'logical history mismatch at step {step}')
    return dict(distributions=len(events), prompt_tokens=len(prompt),
                decode_tokens=len(events)-1)


def export_records(records, base, normalization_tolerance=1e-5, score_tolerance=0.0):
    from compare_distributions import scores as read_scores
    if (type(normalization_tolerance) not in (int, float)
            or not math.isfinite(normalization_tolerance) or not 0 < normalization_tolerance < 1
            or type(score_tolerance) not in (int, float)
            or not math.isfinite(score_tolerance) or score_tolerance < 0):
        raise ValueError('invalid normalization or target-score tolerance')
    base = Path(base).resolve()
    positions, scores, unavailable, seen = [], [], [], set()
    for record in records:
        sample, position, token = record['sample'], record['position'], record['token']
        if not isinstance(sample, str) or not sample.strip():
            raise ValueError('sample must be nonempty')
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
        if not isinstance(descriptor.get('path'), str) or not descriptor['path']:
            raise ValueError('scores must name an array path')
        source = Array(base, descriptor)
        if len(source.shape) not in (1, 2) or not source.shape[-1]:
            raise ValueError('scores must contain a nonempty vocabulary row')
        vocab = source.shape[-1]
        if token >= vocab or any(t >= vocab for t in history):
            raise ValueError('token ID outside score vocabulary')
        values, _ = read_scores(base, record, vocab, normalization_tolerance)
        observed = float(values[token])
        descriptor['path'] = str((base / descriptor['path']).resolve())
        positions.append(dict(sample=sample, position=position, history=history, scores=descriptor))
        target = dict(sample=sample, position=position+1, token=token)
        lp = record['logprob']
        if lp is None:
            if observed != -math.inf:
                raise ValueError('null target logprob requires a zero-probability score')
            unavailable.append(dict(**target, reason='zero target probability'))
        elif type(lp) not in (int, float) or not math.isfinite(lp) or lp > 0:
            raise ValueError('target logprob must be finite and nonpositive or null')
        elif not math.isfinite(observed) or abs(lp-observed) > score_tolerance:
            raise ValueError('target logprob does not match selected score row')
        else:
            scores.append(dict(**target, logprob=lp))
    return dict(positions=positions, scores=scores, unavailable_targets=unavailable)
