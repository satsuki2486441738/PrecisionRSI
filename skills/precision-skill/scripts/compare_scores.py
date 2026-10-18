#!/usr/bin/env python3
"""Compare cache-off full-forward target scores using schema 2 and the standard library.

API: compare(contract_path,side_a_manifest_path,side_b_manifest_path).
Resolve every path relative to its containing JSON.

Contract fields:
  schema_version: 2
  inputs: shared_inputs.json path
  required_coverage: [full_forward_logprobs]
  thresholds: null
  common: {model,weights,tokenizer,config,weight_dtype,autocast,quantization,
           training,no_grad,forward,use_cache,logprob_dtype,log_base,target_shift,
           target_scope,distribution,reduction,percentile}
Model/weights/tokenizer/config identities are {repository:str,revision:str}.
weight_dtype is float16|bfloat16|float32. quantization names the selected scheme
or none. autocast={enabled:bool,target_dtype:dtype_or_null,scope:model_forward}.
Fixed scoring semantics: training=false, no_grad=true, forward=full,
use_cache=false, logprob_dtype=float32, log_base=e, target_shift=-1,
target_scope=assistant_body_nonspecial, distribution=raw_model,
reduction=target_token_weighted, percentile=global_linear_0.99.
Additional cache/SFT/RL required_coverage entries return not_covered.

Manifest fields:
  schema_version: 2; side: nonempty execution-side ID; inputs: JSON path
  observations: executor-format JSONL path; summary: executor-format JSON path
  environment: [JSON paths]; status: [JSON paths]
  expected_ranks: [unique nonnegative integers]; execution: completed
  actual: common fields except target_shift/reduction/percentile
  evidence: {actual_key:[{path:JSON_path,pointer:/JSON/pointer}]}
  uncovered: [coverage names]; side_config: optional object
Side IDs name execution setups; hardware/runtime details belong in environment
records and side_config. The two inputs are peer executions.
Each evidence pointer resolves to an execution value equal to actual[key].
Expected ranks each have an environment and completed status record with rank.
Unknown required identities/settings remain null and return incomplete coverage.

Executor input fields:
  source={dataset,revision,file}, selection, samples
  sample={source_row_1based,prompt_id,messages,rendered_text,input_ids,
          position_ids,attention_mask,score_mask,body_span_half_open,
          target_token_positions,prediction_positions,target_token_ids}
The input format uses complete one-turn messages, positions 0..L-1 and attention
all 1. prompt_id is string or integer; source.local_path locates a resource.
Observations contain case coordinates, target/prediction positions, target IDs,
logprobs,target_count,nll_sum,finite. Summary fields are execution,samples,
target_count,nll_sum,nll_nat_per_token,finite. Logprobs are finite and <=0.

Reports contain execution/comparability/coverage, side NLL diagnostics, paired
absolute NLL difference/MAE/P99 and paired target counts. Available comparisons return no_thresholds;
other statuses are execution_failed,incomparable,not_covered. Metrics use Python
double, coordinate-sorted math.fsum and whole-set linear P99. Summary checks use
64*sys.float_info.epsilon for relative/absolute rounding tolerance. Reusing a
manifest labels self_comparison. Output JSON uses finite literals.
CLI: --contract C --side-a A.json --side-b B.json --output NEW.json
Output paths are created exclusively. Exit: 0=diagnostics, 1=unavailable comparison,
2=CLI/output error.
"""

import argparse
import json
import math
from pathlib import Path
import sys

SCHEMA_VERSION = 2
CORE_VERSION = "3.0.0"
ROUNDING_TOLERANCE = 64 * sys.float_info.epsilon
STATES = ("passed", "exceeded", "incomparable", "not_covered",
          "no_thresholds", "execution_failed")
IDENTITIES = ("model", "weights", "tokenizer", "config")
FIXED = {
    "training": False, "no_grad": True, "forward": "full", "use_cache": False,
    "logprob_dtype": "float32", "log_base": "e", "target_shift": -1,
    "target_scope": "assistant_body_nonspecial", "distribution": "raw_model",
    "reduction": "target_token_weighted", "percentile": "global_linear_0.99",
}
COMMON_KEYS = set(IDENTITIES) | {"weight_dtype", "autocast", "quantization"} | set(FIXED)
CORE_RULES = {key: FIXED[key] for key in ("target_shift", "reduction", "percentile")}
ACTUAL_KEYS = COMMON_KEYS - CORE_RULES.keys()
DTYPES = {"float32", "float16", "bfloat16"}
SAMPLE_FIELDS = ("messages", "rendered_text", "input_ids", "position_ids",
                 "attention_mask", "score_mask", "body_span_half_open",
                 "target_token_positions", "prediction_positions", "target_token_ids")
COVERAGE = ("full_forward_logprobs", "cache", "SFT", "RL")


def _side_id(value):
    return isinstance(value, str) and bool(value.strip())


def _same(a, b):
    """JSON equality without bool/int or int/float coercion."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _ints(value, minimum=0):
    return isinstance(value, list) and all(type(x) is int and x >= minimum for x in value)


def _number(value):
    return type(value) in (int, float)


def _case(row):
    if (not isinstance(row, dict) or type(row.get("source_row_1based")) is not int
            or row["source_row_1based"] < 1
            or type(row.get("prompt_id")) not in (str, int)):
        return None
    # JSON encoding keeps integer and string prompt IDs distinct and sortable.
    return row["source_row_1based"], json.dumps(row["prompt_id"], ensure_ascii=False)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class _Comparison:
    def __init__(self, paths):
        self.paths = {k: str(Path(v).resolve()) for k, v in paths.items()}
        self.blockers = []
        self.evidence_paths = set(self.paths.values())
        self.cache = {}
        self.side_metrics = {}
        self.rule_execution = {key: [] for key in CORE_RULES}

    def issue(self, state, code, where, detail):
        self.blockers.append(dict(state=state, code=code, where=str(where), detail=str(detail)))

    def load(self, owner, path, where, lines=False):
        if not isinstance(path, str) or not path:
            self.issue("not_covered", "missing_path", where, "required artifact path absent")
            return None
        resolved = str((Path(owner).parent / path).resolve())
        self.evidence_paths.add(resolved)
        key = (resolved, lines)
        if key not in self.cache:
            try:
                text = Path(resolved).read_text(encoding="utf-8")
                value = ([json.loads(line, object_pairs_hook=_object)
                          for line in text.splitlines() if line.strip()] if lines
                         else json.loads(text, object_pairs_hook=_object))
                self.cache[key] = (value, None)
            except FileNotFoundError as error:
                self.cache[key] = (None, ("not_covered", "missing_artifact", str(error)))
            except (OSError, UnicodeError, ValueError, RecursionError) as error:
                self.cache[key] = (None, ("execution_failed", "artifact_read", str(error)))
        value, error = self.cache[key]
        if error:
            self.issue(error[0], error[1], where, error[2])
        elif value is None:
            self.issue("not_covered", "null_artifact", where, "null is not observed data or evidence")
        return value

    def mapping(self, value, where):
        if not isinstance(value, dict):
            if value is not None:
                self.issue("incomparable", "object_required", where, "expected JSON object")
            return {}
        return value

    def schema(self, doc, where):
        if type(doc.get("schema_version")) is not int or doc.get("schema_version") != SCHEMA_VERSION:
            self.issue("incomparable", "schema_version", where, "only integer schema_version 2 supported")

    def common(self, value, where, keys=COMMON_KEYS):
        obj = self.mapping(value, where)
        for key in sorted(set(obj) - keys):
            self.issue("incomparable", "unknown_common_field", f"{where}.{key}", "field not accepted in v2")
        for key in sorted(keys):
            v = obj.get(key)
            if v is None:
                self.issue("not_covered", "unknown_condition", where, key)
                continue
            if key in IDENTITIES:
                valid = (isinstance(v, dict) and set(v) == {"repository", "revision"}
                         and all(isinstance(x, str) and x.strip() for x in v.values()))
            elif key == "autocast":
                valid = (isinstance(v, dict) and set(v) == {"enabled", "target_dtype", "scope"}
                         and type(v["enabled"]) is bool and v["scope"] == "model_forward"
                         and ((v["enabled"] and isinstance(v["target_dtype"], str)
                               and v["target_dtype"] in DTYPES)
                              or (not v["enabled"] and v["target_dtype"] is None)))
            elif key in FIXED:
                valid = _same(v, FIXED[key])
            elif key.endswith("dtype"):
                valid = isinstance(v, str) and v in DTYPES
            else:
                valid = isinstance(v, str) and bool(v.strip())
            if not valid:
                self.issue("incomparable", "condition_invalid", where, key)
        return obj

    def evidence(self, manifest, owner, actual, where):
        evidence = self.mapping(manifest.get("evidence"), where + ".evidence")
        for key in sorted(set(evidence) - ACTUAL_KEYS):
            self.issue("incomparable", "unknown_evidence_field", f"{where}.evidence.{key}",
                       "field not accepted in v2")
        for key in sorted(ACTUAL_KEYS):
            refs = evidence.get(key)
            if not isinstance(refs, list) or not refs:
                self.issue("not_covered", "missing_evidence", where, key)
                continue
            for index, ref in enumerate(refs):
                loc = f"{where}.evidence.{key}[{index}]"
                if not isinstance(ref, dict) or set(ref) != {"path", "pointer"}:
                    self.issue("incomparable", "invalid_evidence", loc, "expected path and JSON pointer")
                    continue
                path, pointer = ref["path"], ref["pointer"]
                if isinstance(path, str) and str((Path(owner).parent / path).resolve()) in self.paths.values():
                    self.issue("not_covered", "declaration_not_evidence", loc, "contract/manifest is not execution evidence")
                    continue
                value = self.load(owner, path, loc)
                if value is None:
                    continue
                if not isinstance(pointer, str) or not pointer.startswith("/"):
                    self.issue("incomparable", "invalid_pointer", loc, "nonempty JSON pointer required")
                    continue
                try:
                    for token in pointer[1:].split("/"):
                        # Reject invalid escapes, rather than silently treating them literally.
                        rest = token.replace("~1", "").replace("~0", "")
                        if "~" in rest:
                            raise ValueError("invalid pointer escape")
                        token = token.replace("~1", "/").replace("~0", "~")
                        if isinstance(value, list):
                            if not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                                raise ValueError("invalid array index")
                            value = value[int(token)]
                        else:
                            value = value[token]
                except (KeyError, IndexError, TypeError, ValueError):
                    self.issue("not_covered", "evidence_field_missing", loc, pointer)
                    continue
                if value is None or actual.get(key) is None:
                    self.issue("not_covered", "unknown_evidence", loc, key)
                elif not _same(value, actual[key]):
                    self.issue("incomparable", "evidence_mismatch", loc, key)

    def inputs(self, value, where):
        doc = self.mapping(value, where)
        source = self.mapping(doc.get("source"), where + ".source")
        for key in ("dataset", "revision", "file"):
            if not isinstance(source.get(key), str) or not source[key]:
                self.issue("not_covered", "input_source_missing", where, key)
        if not isinstance(doc.get("selection"), str) or not doc["selection"]:
            self.issue("not_covered", "selection_missing", where, "selection evidence required")
        rows = doc.get("samples")
        if not isinstance(rows, list):
            self.issue("not_covered", "samples_missing", where, "samples array required")
            return {}, doc
        if not rows:
            self.issue("incomparable", "empty_inputs", where, "empty case set")
        samples = {}
        for index, row in enumerate(rows):
            loc = f"{where}.samples[{index}]"
            case = _case(row)
            if case is None:
                self.issue("incomparable", "case_coordinate", loc, "invalid row or prompt_id")
                continue
            if case in samples:
                self.issue("incomparable", "duplicate_case", loc, case)
            samples[case] = row
            missing = [k for k in SAMPLE_FIELDS if row.get(k) is None]
            if missing:
                self.issue("not_covered", "input_fields_missing", loc, missing)
                continue
            ids = row["input_ids"]
            length = len(ids) if isinstance(ids, list) else 0
            arrays = ("input_ids", "position_ids", "attention_mask", "score_mask",
                      "target_token_positions", "prediction_positions", "target_token_ids",
                      "body_span_half_open")
            if not all(_ints(row[k]) for k in arrays):
                self.issue("incomparable", "input_array_type", loc, "integer arrays required (not booleans)")
                continue
            messages = row["messages"]
            valid_messages = (isinstance(messages, list) and len(messages) >= 2
                              and all(isinstance(m, dict) and isinstance(m.get("content"), str)
                                      for m in messages))
            if valid_messages:
                valid_messages = ([m.get("role") for m in messages[-2:]] == ["user", "assistant"]
                                  and all(m.get("role") == "system" for m in messages[:-2])
                                  and bool(messages[-1]["content"]))
            if not valid_messages or not isinstance(row["rendered_text"], str):
                self.issue("incomparable", "history_invalid", loc, "complete selected one-turn history required")
            positions, predictions, targets = (row[k] for k in
                                               ("target_token_positions", "prediction_positions", "target_token_ids"))
            span = row["body_span_half_open"]
            valid = (length > 0 and row["position_ids"] == list(range(length))
                     and row["attention_mask"] == [1] * length
                     and len(row["score_mask"]) == length
                     and all(v in (0, 1) for v in row["score_mask"])
                     and len(span) == 2 and 0 < span[0] < span[1] <= length
                     and len(positions) == len(set(positions))
                     and len(positions) == len(predictions) == len(targets)
                     and set(positions) == {i for i, v in enumerate(row["score_mask"]) if v}
                     and all(span[0] <= p < span[1] for p in positions))
            if valid:
                self.rule_execution["target_shift"].append(loc)
                valid = all(q == p - 1 and t == ids[p]
                            for p, q, t in zip(positions, predictions, targets))
            if not valid:
                self.issue("incomparable", "input_alignment", loc, "length/mask/body/ID/shift mismatch")
            if not positions:
                self.issue("incomparable", "empty_targets", loc, case)
        return samples, doc

    def match_inputs(self, expected, actual, expected_doc, doc, where):
        for key in ("dataset", "revision", "file"):
            a = expected_doc.get("source", {})
            b = doc.get("source", {})
            if (isinstance(a, dict) and isinstance(b, dict)
                    and a.get(key) is not None and b.get(key) is not None
                    and not _same(a[key], b[key])):
                self.issue("incomparable", "input_source_mismatch", where, key)
        if (expected_doc.get("selection") is not None and doc.get("selection") is not None
                and not _same(expected_doc["selection"], doc["selection"])):
            self.issue("incomparable", "selection_mismatch", where, "selection")
        if isinstance(expected_doc.get("samples"), list) and isinstance(doc.get("samples"), list):
            self.sets(set(expected), set(actual), where, "case")
        for case in sorted(set(expected) & set(actual)):
            for field in SAMPLE_FIELDS:
                a, b = expected[case].get(field), actual[case].get(field)
                if a is not None and b is not None and not _same(a, b):
                    self.issue("incomparable", "input_mismatch", where, f"{case}: {field}")

    def sets(self, expected, actual, where, noun):
        for item in sorted(expected - actual):
            self.issue("not_covered", "missing_" + noun, where, item)
        for item in sorted(actual - expected):
            self.issue("incomparable", "extra_" + noun, where, item)

    def ranks(self, manifest, owner, where):
        ranks = manifest.get("expected_ranks")
        if not _ints(ranks) or not ranks or len(ranks) != len(set(ranks)):
            self.issue("incomparable", "expected_ranks_invalid", where, "nonempty unique integer ranks required")
            ranks = []
        execution = manifest.get("execution")
        if execution is None:
            self.issue("not_covered", "execution_missing", where, "execution state unknown")
        elif execution != "completed":
            self.issue("execution_failed", "side_execution", where, execution)
        for kind in ("environment", "status"):
            paths = manifest.get(kind)
            if not isinstance(paths, list):
                self.issue("not_covered", "rank_artifacts_missing", where, kind)
                continue
            seen = set()
            for index, path in enumerate(paths):
                loc = f"{where}.{kind}[{index}]"
                value = self.load(owner, path, loc)
                if value is None:
                    continue
                obj = self.mapping(value, loc)
                rank = obj.get("rank")
                if type(rank) is not int or rank < 0:
                    self.issue("incomparable", "rank_invalid", loc, "integer rank required")
                    continue
                if rank in seen:
                    self.issue("incomparable", "duplicate_rank", loc, rank)
                seen.add(rank)
                if kind == "status":
                    if obj.get("status") is None:
                        self.issue("not_covered", "rank_status_missing", loc, rank)
                    elif obj["status"] != "completed":
                        self.issue("execution_failed", "rank_execution", loc, obj["status"])
            self.sets(set(ranks), seen, where + "." + kind, "rank")

    def finite(self, value, where):
        if not _number(value):
            self.issue("incomparable", "numeric_type", where, repr(value))
            return False
        try:
            valid = math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            self.issue("incomparable", "nonfinite", where, repr(value))
        return valid

    def total(self, values, where):
        try:
            value = math.fsum(values)
        except (OverflowError, ValueError) as error:
            self.issue("execution_failed", "arithmetic_failed", where, error)
            self.issue("incomparable", "nonfinite_result", where, "sum is undefined/overflowed")
            return None
        return value if self.finite(value, where) else None

    def check_summary(self, doc, expected, where):
        for field, value in expected.items():
            if field not in doc:
                self.issue("not_covered", "summary_field_missing", where, field)
                continue
            saved = doc[field]
            if type(value) is int:
                valid = type(saved) is int and saved == value
            else:
                valid = self.finite(saved, where + "." + field)
                if valid:
                    valid = math.isclose(saved, value, rel_tol=ROUNDING_TOLERANCE,
                                         abs_tol=ROUNDING_TOLERANCE)
            if not valid:
                self.issue("incomparable", "summary_mismatch", where, field)
        if doc.get("finite") is not True:
            self.issue("not_covered" if "finite" not in doc else "incomparable",
                       "finite_flag", where, "finite must be true; values independently checked")

    def observations(self, rows, samples, summary, where):
        expected, known_cases = {}, set()
        for case, sample in samples.items():
            arrays = [sample.get(k) for k in ("target_token_positions", "prediction_positions", "target_token_ids")]
            if all(_ints(x) for x in arrays) and len({len(x) for x in arrays}) == 1:
                known_cases.add(case)
                for p, q, t in zip(*arrays):
                    expected[case + (p,)] = (q, t)
        observed, seen_cases = {}, set()
        if rows is None:
            self.sets(set(expected), set(), where, "token")
            return {}
        for index, row in enumerate(rows):
            loc = f"{where}.observations[{index}]"
            case = _case(row)
            if case is None:
                self.issue("incomparable", "case_coordinate", loc, "invalid observation case")
                continue
            if case in seen_cases:
                self.issue("incomparable", "duplicate_case", loc, case)
            seen_cases.add(case)
            arrays = [row.get(k) for k in ("target_token_positions", "prediction_positions", "target_token_ids")]
            values = row.get("logprobs")
            if any(x is None for x in arrays) or values is None:
                self.issue("not_covered", "observation_fields_missing", loc, case)
                continue
            if not all(_ints(x) for x in arrays) or not isinstance(values, list):
                self.issue("incomparable", "observation_array_type", loc, "invalid integer/value arrays")
                continue
            if len({len(x) for x in arrays + [values]}) != 1:
                self.issue("incomparable", "observation_lengths", loc, "array lengths differ")
                continue
            local = {}
            all_valid = True
            for p, q, t, value in zip(*arrays, values):
                key = case + (p,)
                if key in observed:
                    self.issue("incomparable", "duplicate_token", loc, key)
                observed[key] = value
                local[key] = value
                if key in expected and expected[key] != (q, t):
                    self.issue("incomparable", "target_alignment", loc, key)
                valid = self.finite(value, f"{loc}.{key}")
                if valid and value > 0:
                    self.issue("incomparable", "positive_logprob", f"{loc}.{key}",
                               f"raw_model natural logprob must be <= 0; got {value!r}")
                    valid = False
                all_valid = valid and all_valid
            expected_summary = {"target_count": len(values)}
            if all_valid:
                total = self.total([-local[k] for k in sorted(local)], loc)
                if total is not None:
                    expected_summary["nll_sum"] = total
            self.check_summary(row, expected_summary, loc)
        if samples:
            self.sets(set(samples), seen_cases, where + ".observations", "case")
        # A missing target array is unknown coverage, not evidence that its
        # observed tokens are extras. Still check every known coordinate.
        checkable = {k for k in observed if k[:2] in known_cases or (samples and k[:2] not in samples)}
        self.sets(set(expected), checkable, where, "token")
        if not observed:
            self.issue("incomparable", "empty_observations", where, "empty scoring set")
        # Raw diagnostics remain available when identities are unknown, but never
        # represent a partial paired metric. Duplicate/malformed rows block them.
        local_errors = any(b["where"].startswith(where) and b["state"] != "no_thresholds"
                           and b["code"] in {"duplicate_case", "duplicate_token", "numeric_type", "nonfinite", "positive_logprob",
                                             "observation_lengths", "observation_array_type", "case_coordinate"}
                           for b in self.blockers)
        if observed and not local_errors:
            total = self.total([-observed[k] for k in sorted(observed)], where)
            if total is not None:
                metrics = {"samples": len(seen_cases), "target_count": len(observed),
                           "nll_sum": total, "nll_nat_per_token": total / len(observed)}
                self.side_metrics[where] = metrics
                self.rule_execution["reduction"].append(where)
                if summary is not None:
                    obj = self.mapping(summary, where + ".summary")
                    self.check_summary(obj, metrics, where + ".summary")
        if isinstance(summary, dict):
            if summary.get("execution") is None:
                self.issue("not_covered", "summary_execution_missing", where, "execution missing")
            elif summary["execution"] != "completed":
                self.issue("execution_failed", "summary_execution", where, summary["execution"])
        return observed

    def run(self):
        docs = {k: self.mapping(self.load(v, v, k), k) for k, v in self.paths.items()}
        for key, doc in docs.items():
            self.schema(doc, key)
        contract = docs["contract"]
        common = self.common(contract.get("common"), "contract.common")
        if "thresholds" not in contract:
            self.issue("not_covered", "thresholds_missing", "contract", "explicit thresholds:null required")
        elif contract["thresholds"] is not None:
            self.issue("incomparable", "thresholds_unsupported", "contract", "v2 supports only null thresholds")
        self.issue("no_thresholds", "no_thresholds", "contract", "diagnostic only; never passed/exceeded")
        required = contract.get("required_coverage")
        if (not isinstance(required, list) or not all(isinstance(x, str) and x in COVERAGE for x in required)
                or len(required) != len(set(required)) or "full_forward_logprobs" not in required):
            self.issue("incomparable", "required_coverage_invalid", "contract", "unique known coverage names including full_forward_logprobs required")
            required = ["full_forward_logprobs"]
        for item in required:
            if item != "full_forward_logprobs":
                self.issue("not_covered", "unsupported_coverage", "contract", item)
        shared = self.load(self.paths["contract"], contract.get("inputs"), "contract.inputs")
        expected, shared_doc = self.inputs(shared, "contract.inputs")
        scores = {}
        for side in ("side_a", "side_b"):
            manifest, owner = docs[side], self.paths[side]
            if not _side_id(manifest.get("side")):
                self.issue("incomparable", "side_identity", side, "nonempty execution-side ID required")
            actual = self.common(manifest.get("actual"), side + ".actual", ACTUAL_KEYS)
            for key in sorted(ACTUAL_KEYS):
                if common.get(key) is not None and actual.get(key) is not None and not _same(common[key], actual[key]):
                    self.issue("incomparable", "condition_mismatch", side, key)
            self.evidence(manifest, owner, actual, side)
            self.ranks(manifest, owner, side)
            uncovered = manifest.get("uncovered")
            if not isinstance(uncovered, list) or not all(isinstance(x, str) for x in uncovered):
                self.issue("not_covered", "coverage_declaration_missing", side, "uncovered list required")
            else:
                for item in sorted(set(uncovered) & set(required)):
                    self.issue("not_covered", "declared_uncovered", side, item)
            inputs, input_doc = self.inputs(self.load(owner, manifest.get("inputs"), side + ".inputs"), side + ".inputs")
            self.match_inputs(expected, inputs, shared_doc, input_doc, side + ".inputs")
            rows = self.load(owner, manifest.get("observations"), side + ".observations", lines=True)
            summary = self.load(owner, manifest.get("summary"), side + ".summary")
            # Validate saved observations against whatever input coordinates are
            # actually available. Missing canonical inputs already block pairing;
            # they must not turn known side-local tokens into spurious extras.
            score_inputs = inputs if not isinstance(shared_doc.get("samples"), list) else expected
            scores[side] = self.observations(rows, score_inputs, summary, side)
        metrics = None
        if not any(b["state"] != "no_thresholds" for b in self.blockers):
            keys = sorted(scores["side_a"])
            errors = [abs(scores["side_b"][k] - scores["side_a"][k]) for k in keys]
            if all([self.finite(v, f"metrics.error[{i}]") for i, v in enumerate(errors)]):
                total = self.total(errors, "metrics.mae")
                gap = abs(self.side_metrics["side_b"]["nll_nat_per_token"] - self.side_metrics["side_a"]["nll_nat_per_token"])
                valid_gap = self.finite(gap, "metrics.nll_abs_diff")
                if total is not None and valid_gap:
                    errors.sort()
                    h = 0.99 * (len(errors) - 1)
                    j, k = math.floor(h), math.ceil(h)
                    p99 = errors[j] + (h - j) * (errors[k] - errors[j])
                    self.rule_execution["percentile"].append("metrics.p99")
                    if self.finite(p99, "metrics.p99"):
                        metrics = {"target_count": len(keys),
                                   "nll_abs_diff_nat_per_token": gap, "mae_nat": total / len(keys),
                                   "p99_nat": p99, "p99_method": "global linear h=0.99*(N-1)"}
        states = {b["state"] for b in self.blockers}
        status = next(s for s in ("execution_failed", "incomparable", "not_covered", "no_thresholds") if s in states)
        coverage = {x: "not_covered" for x in COVERAGE}
        coverage["full_forward_logprobs"] = status
        blockers = sorted(self.blockers, key=lambda b: (b["where"], b["code"], b["detail"], b["state"]))
        same_observations = False
        r, c = docs["side_a"].get("observations"), docs["side_b"].get("observations")
        if isinstance(r, str) and isinstance(c, str):
            same_observations = ((Path(self.paths["side_a"]).parent / r).resolve()
                                 == (Path(self.paths["side_b"]).parent / c).resolve())
        return {"schema_version": SCHEMA_VERSION, "core_version": CORE_VERSION, "paths": self.paths,
                "evidence_paths": sorted(self.evidence_paths), "self_comparison": same_observations,
                "sides": {k: docs[k].get("side") if _side_id(docs[k].get("side")) else None
                          for k in ("side_a", "side_b")},
                "status": status, "execution": "execution_failed" if "execution_failed" in states else "completed",
                "comparability": "incomparable" if "incomparable" in states else
                                 ("unknown" if states & {"not_covered", "execution_failed"} else "comparable"),
                "numerical_status": "no_thresholds" if metrics is not None else "undefined",
                "coverage": coverage, "required_coverage": required, "blockers": blockers,
                "side_metrics": self.side_metrics, "metrics": metrics,
                "rules": {key: {"definition": value,
                                "declaration_valid": _same(common.get(key), value),
                                "executed_at": sorted(self.rule_execution[key])}
                          for key, value in CORE_RULES.items()},
                "summary_rounding_tolerance": {"relative": ROUNDING_TOLERANCE, "absolute": ROUNDING_TOLERANCE,
                                               "scope": "summary reconciliation only; not acceptance thresholds"}}


def compare(contract, side_a, side_b):
    """Compare three filesystem paths; return a JSON-serializable diagnostic report."""
    return _Comparison(dict(contract=contract, side_a=side_a, side_b=side_b)).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("contract", "side_a", "side_b", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Reserve before computing; never overwrite any existing artifact.
        with args.output.open("x", encoding="utf-8") as stream:
            report = compare(args.contract, args.side_a, args.side_b)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as error:
        print(f"compare_scores: {error}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "no_thresholds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
