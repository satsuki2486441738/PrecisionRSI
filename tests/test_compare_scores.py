"""Permanent synthetic schema-v2 tests; all artifacts live in TemporaryDirectory."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "precision-skill" / "scripts" / "compare_scores.py"
SPEC = importlib.util.spec_from_file_location("compare_scores", SCRIPT)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def sample(row, count):
    ids = list(range(10, 12 + count))
    positions = list(range(1, 1 + count))
    return {"source_row_1based": row, "prompt_id": f"p{row}",
            "messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": f"question {row}"},
                         {"role": "assistant", "content": "body"}],
            "rendered_text": "system user body end", "input_ids": ids,
            "position_ids": list(range(len(ids))), "attention_mask": [1] * len(ids),
            "score_mask": [0] + [1] * count + [0],
            "body_span_half_open": [1, 1 + count], "target_token_positions": positions,
            "prediction_positions": [p - 1 for p in positions],
            "target_token_ids": [ids[p] for p in positions]}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compare-scores-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.common = dict(CORE.FIXED)
        self.common.update({k: {"repository": "test/model", "revision": "immutable-test-revision"}
                            for k in CORE.IDENTITIES})
        self.common.update(weight_dtype="bfloat16", autocast={"enabled": False, "target_dtype": None, "scope": "model_forward"}, quantization="none")
        self.inputs = {"source": {"dataset": "test/data", "revision": "r1", "file": "train.jsonl",
                                  "local_path": "/irrelevant/old/location"},
                       "selection": "synthetic complete single-turn cases",
                       "samples": [sample(1, 1), sample(2, 3)]}
        self.write("inputs.json", self.inputs)
        self.contract = {"schema_version": 2, "inputs": "inputs.json", "common": self.common,
                         "required_coverage": ["full_forward_logprobs"],
                         "thresholds": None}
        self.write("contract.json", self.contract)
        self.manifests = {}
        self.make_side("side_a", [-1.0, -2.0, -3.0, -4.0], "GPU")
        self.make_side("side_b", [-1.0, -2.0, -3.0, -4.0], "NPU")

    def write(self, name, obj):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def make_side(self, side, values, device=None):
        self.write(f"{side}/inputs.json", self.inputs)
        rows, cursor = [], 0
        for s in self.inputs["samples"]:
            n = len(s["target_token_positions"])
            v = values[cursor:cursor + n]
            cursor += n
            row = {key: copy.deepcopy(s[key]) for key in
                   ("source_row_1based", "prompt_id", "target_token_positions", "prediction_positions", "target_token_ids")}
            row.update(logprobs=v, finite=True, target_count=n, nll_sum=-math.fsum(v))
            rows.append(row)
        self.write_rows(side, rows)
        total = -math.fsum(values)
        self.write(f"{side}/summary.json", {"execution": "completed", "samples": len(rows),
                                           "target_count": len(values), "nll_sum": total,
                                           "nll_nat_per_token": total / len(values), "finite": True})
        for rank in (0, 1):
            self.write(f"{side}/env{rank}.json", {"rank": rank, "device": device or side,
                                                 "observed": self.common})
            self.write(f"{side}/status{rank}.json", {"rank": rank, "status": "completed"})
        manifest = {"schema_version": 2, "side": device or self.manifests[side]["side"],
                    "inputs": "inputs.json", "observations": "scores.jsonl", "summary": "summary.json",
                    "environment": ["env0.json", "env1.json"], "status": ["status0.json", "status1.json"],
                    "expected_ranks": [0, 1], "execution": "completed", "actual": {k: copy.deepcopy(self.common[k]) for k in CORE.ACTUAL_KEYS},
                    "evidence": {k: [{"path": "env0.json", "pointer": "/observed/" + k},
                                     {"path": "env1.json", "pointer": "/observed/" + k}]
                                 for k in CORE.ACTUAL_KEYS},
                    "uncovered": ["cache", "SFT", "RL"], "side_config": {"framework": "synthetic"}}
        self.manifests[side] = manifest
        self.write(f"{side}/manifest.json", manifest)

    def rows(self, side="side_b"):
        return [json.loads(s) for s in (self.root / side / "scores.jsonl").read_text().splitlines()]

    def write_rows(self, side, rows):
        path = self.root / side / "scores.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def compare(self, side_b="side_b/manifest.json"):
        return CORE.compare(self.root / "contract.json", self.root / "side_a/manifest.json", self.root / side_b)

    def codes(self, report):
        return {b["code"] for b in report["blockers"]}

    def assertBlocked(self, report, state, code):
        self.assertEqual(report["status"], state)
        self.assertIn(code, self.codes(report))
        self.assertIsNone(report["metrics"])
        self.assertIn("no_thresholds", self.codes(report))
        json.dumps(report, allow_nan=False)

    def test_task_defined_sides_and_swap_symmetry(self):
        for labels in [('cpu-baseline', 'accelerator-trial'), ('gpu-stack-a', 'gpu-stack-b'),
                       ('side_a', 'side_b'), ('NPU', 'GPU')]:
            with self.subTest(labels=labels):
                self.write('contract.json', self.contract)
                self.make_side('side_a', [-1., -2., -3., -4.], labels[0])
                self.make_side('side_b', [-2., -3., -4., -5.], labels[1])
                report = self.compare()
                self.assertEqual(report['status'], 'no_thresholds')
                self.assertEqual(report['sides'], dict(zip(('side_a', 'side_b'), labels)))
                self.assertAlmostEqual(report['metrics']['nll_abs_diff_nat_per_token'], 1.)
                self_check = self.compare('side_a/manifest.json')
                self.assertTrue(self_check['self_comparison'])
                self.assertEqual(self_check['status'], 'no_thresholds')
                self.write('contract.json', self.contract)
                reverse = CORE.compare(self.root / 'contract.json', self.root / 'side_b/manifest.json',
                                       self.root / 'side_a/manifest.json')
                self.assertEqual(reverse['status'], 'no_thresholds')
                self.assertEqual(report['metrics'], reverse['metrics'])
                self.assertEqual(reverse['sides'], {'side_a': labels[1], 'side_b': labels[0]})
                self.assertAlmostEqual(reverse['metrics']['nll_abs_diff_nat_per_token'], 1.)

    def test_invalid_side_identifiers(self):
        for side in ('side_a', 'side_b'):
            for label in ['', '   ', None, 1, ['trial']]:
                with self.subTest(side=side, label=label):
                    original = self.manifests[side]['side']
                    self.manifests[side]['side'] = label
                    self.write(side+'/manifest.json', self.manifests[side])
                    self.assertBlocked(self.compare(), 'incomparable', 'side_identity')
                    self.manifests[side]['side'] = original
                    self.write(side+'/manifest.json', self.manifests[side])

    def test_self_comparison_zero_is_not_pass(self):
        report = self.compare("side_a/manifest.json")
        self.assertTrue(report["self_comparison"])
        self.assertEqual(report["sides"], {"side_a": "GPU", "side_b": "GPU"})
        self.assertEqual(report["status"], "no_thresholds")
        self.assertEqual(report["comparability"], "comparable")
        self.assertEqual(report["metrics"]["nll_abs_diff_nat_per_token"], 0)
        self.assertEqual(report["metrics"]["mae_nat"], 0)
        self.assertEqual(report["metrics"]["p99_nat"], 0)
        self.assertEqual(report["side_metrics"]["side_a"]["nll_nat_per_token"], 2.5)
        self.assertEqual(report["coverage"]["cache"], "not_covered")
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(set(CORE.STATES), {"passed", "exceeded", "incomparable", "not_covered", "no_thresholds", "execution_failed"})

    def test_weighted_absolute_difference_and_global_percentile(self):
        self.make_side("side_b", [-2.0, -4.0, -6.0, -8.0])
        report = self.compare()
        self.assertFalse(report["self_comparison"])
        m = report["metrics"]
        self.assertEqual(m["target_count"], 4)
        self.assertEqual(report["side_metrics"]["side_b"]["nll_nat_per_token"], 5.0)
        self.assertEqual(m["nll_abs_diff_nat_per_token"], 2.5)
        self.assertEqual(m["mae_nat"], 2.5)
        self.assertAlmostEqual(m["p99_nat"], 3.97)

    def test_cancellation_keeps_absolute_errors(self):
        self.make_side("side_b", [-0.5, -2.5, -2.5, -4.5])
        m = self.compare()["metrics"]
        self.assertEqual(m["nll_abs_diff_nat_per_token"], 0)
        self.assertEqual(m["mae_nat"], 0.5)
        self.assertEqual(m["p99_nat"], 0.5)

    def test_single_element(self):
        self.inputs["samples"] = [sample(1, 1)]
        self.write("inputs.json", self.inputs)
        self.make_side("side_a", [-1.0])
        self.make_side("side_b", [-0.25])
        m = self.compare()["metrics"]
        self.assertEqual(m["p99_nat"], 0.75)
        self.assertEqual(m["mae_nat"], 0.75)
        self.assertEqual(m["nll_abs_diff_nat_per_token"], 0.75)

    def test_reordering_records_tokens_inputs_and_ranks(self):
        before = self.compare()
        rows = self.rows()
        rows.reverse()
        for row in rows:
            for key in ("target_token_positions", "prediction_positions", "target_token_ids", "logprobs"):
                row[key].reverse()
        self.write_rows("side_b", rows)
        data = self.read("side_b/inputs.json")
        data["samples"].reverse()
        data["source"]["local_path"] = "/new/location"
        self.write("side_b/inputs.json", data)
        man = self.manifests["side_b"]
        for key in ("environment", "status", "expected_ranks"):
            man[key].reverse()
        self.write("side_b/manifest.json", man)
        after = self.compare()
        self.assertEqual(before["metrics"], after["metrics"])
        self.assertEqual(after, self.compare())
        self.assertEqual(before["blockers"], after["blockers"])

    def test_duplicate_case_and_token(self):
        rows = self.rows()
        rows.append(copy.deepcopy(rows[0]))
        self.write_rows("side_b", rows)
        r = self.compare()
        self.assertBlocked(r, "incomparable", "duplicate_case")
        self.assertIn("duplicate_token", self.codes(r))

    def test_duplicate_position_in_one_record(self):
        rows = self.rows()
        rows[1]["target_token_positions"][1] = rows[1]["target_token_positions"][0]
        self.write_rows("side_b", rows)
        r = self.compare()
        self.assertBlocked(r, "incomparable", "duplicate_token")
        self.assertIn("missing_token", self.codes(r))

    def test_missing_case_is_uncovered_not_intersection(self):
        self.write_rows("side_b", self.rows()[:1])
        summary = self.read("side_b/summary.json")
        summary.update(samples=1, target_count=1, nll_sum=1.0, nll_nat_per_token=1.0)
        self.write("side_b/summary.json", summary)
        self.assertBlocked(self.compare(), "not_covered", "missing_case")

    def test_missing_token_is_uncovered(self):
        rows = self.rows()
        for key in ("target_token_positions", "prediction_positions", "target_token_ids", "logprobs"):
            rows[1][key].pop()
        rows[1].update(target_count=2, nll_sum=5.0)
        self.write_rows("side_b", rows)
        summary = self.read("side_b/summary.json")
        summary.update(target_count=3, nll_sum=6.0, nll_nat_per_token=2.0)
        self.write("side_b/summary.json", summary)
        self.assertBlocked(self.compare(), "not_covered", "missing_token")

    def test_extra_case_and_token(self):
        rows = self.rows()
        extra = copy.deepcopy(rows[0])
        extra["source_row_1based"] = 3
        rows.append(extra)
        self.write_rows("side_b", rows)
        report = self.compare()
        self.assertBlocked(report, "incomparable", "extra_case")
        self.assertIn("extra_token", self.codes(report))

    def test_identity_and_common_condition_conflicts(self):
        for key, value in (("model", {"repository": "different", "revision": "r2"}),
                           ("training", True), ("weight_dtype", "float32"),
                           ("no_grad", False), ("quantization", "int8"), ("use_cache", True)):
            with self.subTest(key=key):
                man = copy.deepcopy(self.manifests["side_b"])
                man["actual"][key] = value
                self.write("side_b/manifest.json", man)
                self.assertBlocked(self.compare(), "incomparable", "condition_mismatch")

    def test_input_history_masks_ids_positions_and_body_conflicts(self):
        mutations = {
            "messages": lambda s: s["messages"][0].update(content="different system history"),
            "rendered_text": lambda s: s.update(rendered_text="different rendering"),
            "input_ids": lambda s: s["input_ids"].__setitem__(0, 900),
            "position_ids": lambda s: s["position_ids"].__setitem__(0, 1),
            "attention_mask": lambda s: s["attention_mask"].__setitem__(1, 0),
            "score_mask": lambda s: s["score_mask"].__setitem__(1, 0),
            "body_span_half_open": lambda s: s.update(body_span_half_open=[0, 2]),
            "prediction_positions": lambda s: s["prediction_positions"].__setitem__(0, 1),
            "target_token_ids": lambda s: s["target_token_ids"].__setitem__(0, 777),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                data = copy.deepcopy(self.inputs)
                mutate(data["samples"][0])
                self.write("side_b/inputs.json", data)
                self.assertBlocked(self.compare(), "incomparable", "input_mismatch")

    def test_observation_shift_and_target_id(self):
        for field in ("prediction_positions", "target_token_ids"):
            with self.subTest(field=field):
                rows = self.rows("side_a")
                rows[0][field][0] += 1
                self.write_rows("side_b", rows)
                self.assertBlocked(self.compare(), "incomparable", "target_alignment")

    def test_bad_array_lengths(self):
        rows = self.rows()
        rows[0]["logprobs"].append(-2.0)
        self.write_rows("side_b", rows)
        self.assertBlocked(self.compare(), "incomparable", "observation_lengths")

    def test_empty_observations(self):
        self.write_rows("side_b", [])
        r = self.compare()
        self.assertBlocked(r, "incomparable", "empty_observations")
        self.assertIn("missing_token", self.codes(r))

    def test_empty_contract_inputs(self):
        self.inputs["samples"] = []
        self.write("inputs.json", self.inputs)
        self.assertBlocked(self.compare(), "incomparable", "empty_inputs")

    def test_nonfinite_values_despite_true_flags(self):
        for v in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=repr(v)):
                rows = self.rows("side_a")
                rows[0]["logprobs"][0] = v
                self.write_rows("side_b", rows)
                report = self.compare()
                self.assertBlocked(report, "incomparable", "nonfinite")
                self.assertNotIn("side_b", report["side_metrics"])

    def test_positive_logprobs_on_either_side_are_invalid(self):
        for side in ("side_a", "side_b"):
            for values in ([1, 2, 3, 4], [math.nextafter(0.0, math.inf), -2.0, -3.0, -4.0]):
                with self.subTest(side=side, values=values):
                    self.make_side("side_a", [-1.0, -2.0, -3.0, -4.0])
                    self.make_side("side_b", [-1.0, -2.0, -3.0, -4.0])
                    self.make_side(side, values)
                    report = self.compare()
                    self.assertBlocked(report, "incomparable", "positive_logprob")
                    self.assertNotIn(side, report["side_metrics"])
                    self.assertEqual(report["comparability"], "incomparable")
                    self.assertEqual(report["numerical_status"], "undefined")
                    self.assertEqual(sum(b["code"] == "positive_logprob" for b in report["blockers"]),
                                     sum(value > 0 for value in values))

    def test_zero_and_negative_logprobs_remain_valid(self):
        for values in ([0.0, -0.0, 0, 0], [-1.0, -2.0, -3.0, -4.0],
                       [math.nextafter(0.0, -math.inf), -2.0, -3.0, -4.0]):
            with self.subTest(values=values):
                self.make_side("side_a", values)
                self.make_side("side_b", values)
                report = self.compare()
                self.assertEqual(report["status"], "no_thresholds")
                self.assertEqual(report["comparability"], "comparable")
                self.assertEqual(report["metrics"]["mae_nat"], 0.0)
                for metrics in report["side_metrics"].values():
                    self.assertGreaterEqual(metrics["nll_nat_per_token"], 0.0)

    def test_positive_logprob_keeps_concurrent_blockers(self):
        self.make_side("side_b", [1, 2, 3, 4])
        rows = self.rows()
        rows[1]["logprobs"][1] = float("nan")
        rows[1]["prediction_positions"][0] += 1
        self.write_rows("side_b", rows)
        self.write("side_b/status1.json", {"rank": 1, "status": "execution_failed"})
        report = self.compare()
        self.assertBlocked(report, "execution_failed", "positive_logprob")
        self.assertTrue({"nonfinite", "target_alignment", "rank_execution", "no_thresholds"}
                        <= self.codes(report))
        self.assertNotIn("side_b", report["side_metrics"])
        self.assertEqual(report["comparability"], "incomparable")

    def test_sum_overflow_is_incomparable_and_execution_failed(self):
        rows = self.rows()
        rows[1]["logprobs"] = [-1e308] * 3
        self.write_rows("side_b", rows)
        r = self.compare()
        self.assertBlocked(r, "execution_failed", "arithmetic_failed")
        self.assertIn("nonfinite_result", self.codes(r))
        self.assertEqual(r["comparability"], "incomparable")

    def test_difference_overflow(self):
        self.inputs["samples"] = [sample(1, 1)]
        self.write("inputs.json", self.inputs)
        self.make_side("side_a", [-1e308])
        self.make_side("side_b", [1e308])
        # Reject the illegal positive operand before attempting paired subtraction.
        self.assertBlocked(self.compare(), "incomparable", "positive_logprob")

    def test_incorrect_summary_and_per_row_summary(self):
        for filename, field, value in (("side_b/summary.json", "samples", 3),
                                       ("side_b/summary.json", "target_count", 5),
                                       ("side_b/summary.json", "nll_sum", 11.0),
                                       ("side_b/summary.json", "nll_nat_per_token", 9.0)):
            with self.subTest(field=field):
                original = self.read(filename)
                edited = dict(original, **{field: value})
                self.write(filename, edited)
                self.assertBlocked(self.compare(), "incomparable", "summary_mismatch")
                self.write(filename, original)
        rows = self.rows()
        rows[0]["nll_sum"] = 20.0
        self.write_rows("side_b", rows)
        self.assertBlocked(self.compare(), "incomparable", "summary_mismatch")

    def test_summary_rounding_is_not_a_business_threshold(self):
        summary = self.read("side_b/summary.json")
        summary["nll_sum"] = math.nextafter(summary["nll_sum"], math.inf)
        self.write("side_b/summary.json", summary)
        r = self.compare()
        self.assertEqual(r["status"], "no_thresholds")
        self.assertIn("not acceptance", r["summary_rounding_tolerance"]["scope"])

    def test_bad_json_and_duplicate_json_keys(self):
        for text in ("{bad json", '{"rank":0,"rank":1}'):
            with self.subTest(text=text):
                (self.root / "side_b/env0.json").write_text(text)
                self.assertBlocked(self.compare(), "execution_failed", "artifact_read")

    def test_bad_jsonl_and_missing_artifact(self):
        path = self.root / "side_b/scores.jsonl"
        path.write_text('{"broken":\n')
        self.assertBlocked(self.compare(), "execution_failed", "artifact_read")
        path.unlink()
        self.assertBlocked(self.compare(), "not_covered", "missing_artifact")

    def test_rank_failure_missing_and_duplicate(self):
        self.write("side_b/status0.json", {"rank": 0, "status": "execution_failed"})
        (self.root / "side_b/status1.json").unlink()
        r = self.compare()
        self.assertBlocked(r, "execution_failed", "rank_execution")
        self.assertIn("missing_rank", self.codes(r))
        self.write("side_b/status0.json", {"rank": 0, "status": "completed"})
        self.write("side_b/status1.json", {"rank": 0, "status": "completed"})
        self.assertBlocked(self.compare(), "incomparable", "duplicate_rank")

    def test_missing_rank_only_is_uncovered(self):
        (self.root / "side_b/env1.json").unlink()
        self.assertBlocked(self.compare(), "not_covered", "missing_rank")

    def test_multiple_independent_blockers_preserved(self):
        rows = self.rows()
        rows[0]["logprobs"][0] = float("nan")
        rows.pop()
        self.write_rows("side_b", rows)
        self.write("side_b/status1.json", {"rank": 1, "status": "execution_failed"})
        (self.root / "side_b/env0.json").write_text("invalid JSON")
        r = self.compare()
        self.assertBlocked(r, "execution_failed", "artifact_read")
        self.assertTrue({"nonfinite", "missing_token", "rank_execution", "no_thresholds"} <= self.codes(r))
        self.assertEqual(r["comparability"], "incomparable")
        self.assertEqual(r["blockers"], sorted(r["blockers"], key=lambda b: (b["where"], b["code"], b["detail"], b["state"])))

    def test_unknown_identities_block_but_raw_nll_is_available(self):
        man = self.manifests["side_b"]
        man["actual"]["tokenizer"] = None
        man["actual"]["config"] = None
        man["evidence"].pop("config")
        self.write("side_b/manifest.json", man)
        r = self.compare()
        self.assertBlocked(r, "not_covered", "unknown_condition")
        self.assertIn("missing_evidence", self.codes(r))
        self.assertEqual(r["side_metrics"]["side_b"]["nll_nat_per_token"], 2.5)

    def test_missing_and_false_evidence_not_verified_strings(self):
        for refs, code, state in (("verified", "missing_evidence", "not_covered"),
                                  ([{"path": "env0.json", "pointer": "/absent"}], "evidence_field_missing", "not_covered"),
                                  ([{"path": "env0.json", "pointer": "/rank"}], "evidence_mismatch", "incomparable"),
                                  ([{"path": "manifest.json", "pointer": "/actual/model"}], "declaration_not_evidence", "not_covered"),
                                  ([{"path": "absent.json", "pointer": "/model"}], "missing_artifact", "not_covered")):
            with self.subTest(code=code):
                man = copy.deepcopy(self.manifests["side_b"])
                man["evidence"]["model"] = refs
                self.write("side_b/manifest.json", man)
                self.assertBlocked(self.compare(), state, code)

    def test_unknown_contract_condition_not_inferred(self):
        self.contract["common"]["config"] = None
        self.write("contract.json", self.contract)
        self.assertBlocked(self.compare(), "not_covered", "unknown_condition")

    def test_thresholds_nonnull_explicitly_unsupported(self):
        for value in ({}, {"mae": 1e-3}, 0, False):
            with self.subTest(value=value):
                self.contract["thresholds"] = value
                self.write("contract.json", self.contract)
                self.assertBlocked(self.compare(), "incomparable", "thresholds_unsupported")

    def test_missing_thresholds_is_not_an_implicit_null(self):
        del self.contract["thresholds"]
        self.write("contract.json", self.contract)
        self.assertBlocked(self.compare(), "not_covered", "thresholds_missing")

    def test_required_unimplemented_coverage(self):
        for value in ("cache", "SFT", "RL"):
            with self.subTest(value=value):
                self.contract["required_coverage"] = ["full_forward_logprobs", value]
                self.write("contract.json", self.contract)
                self.assertBlocked(self.compare(), "not_covered", "unsupported_coverage")

    def test_declared_missing_required_coverage(self):
        man = self.manifests["side_b"]
        man["uncovered"].append("full_forward_logprobs")
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "not_covered", "declared_uncovered")

    def test_no_old_schema_or_boolean_schema(self):
        for version in (0, 1, 3, True, 2.0, "2", None):
            with self.subTest(version=version):
                man = copy.deepcopy(self.manifests["side_b"])
                man["schema_version"] = version
                self.write("side_b/manifest.json", man)
                self.assertBlocked(self.compare(), "incomparable", "schema_version")

    def test_boolean_not_integer_or_number(self):
        for field in ("source_row_1based", "prompt_id", "target_count", "nll_sum"):
            with self.subTest(field=field):
                rows = self.rows("side_a")
                rows[0][field] = True
                self.write_rows("side_b", rows)
                self.assertEqual(self.compare()["status"], "incomparable")
        rows = self.rows("side_a")
        rows[0]["logprobs"] = [True]
        self.write_rows("side_b", rows)
        self.assertBlocked(self.compare(), "incomparable", "numeric_type")

    def test_boolean_in_input_arrays_and_ranks(self):
        data = copy.deepcopy(self.inputs)
        data["samples"][0]["attention_mask"][0] = True
        self.write("side_b/inputs.json", data)
        self.assertBlocked(self.compare(), "incomparable", "input_array_type")
        self.write("side_b/inputs.json", self.inputs)
        man = self.manifests["side_b"]
        man["expected_ranks"] = [False, True]
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "incomparable", "expected_ranks_invalid")

    def test_invalid_container_types_do_not_crash(self):
        for field, value in (("actual", []), ("expected_ranks", {}), ("evidence", False),
                             ("status", "status0.json"), ("uncovered", {})):
            with self.subTest(field=field):
                man = copy.deepcopy(self.manifests["side_b"])
                man[field] = value
                self.write("side_b/manifest.json", man)
                r = self.compare()
                self.assertIn(r["status"], ("incomparable", "not_covered"))
                json.dumps(r, allow_nan=False)

    def test_missing_input_fields_and_canonical_artifact_are_uncovered(self):
        for name in ("inputs.json", "side_b/inputs.json"):
            for field in CORE.SAMPLE_FIELDS:
                with self.subTest(file=name, field=field):
                    data = copy.deepcopy(self.inputs)
                    del data["samples"][0][field]
                    self.write(name, data)
                    self.assertBlocked(self.compare(), "not_covered", "input_fields_missing")
                    self.write(name, self.inputs)
        (self.root / "inputs.json").unlink()
        self.assertBlocked(self.compare(), "not_covered", "missing_artifact")

    def test_null_summary_and_evidence_are_uncovered(self):
        summary = self.read("side_b/summary.json")
        self.write("side_b/summary.json", None)
        self.assertBlocked(self.compare(), "not_covered", "null_artifact")
        self.write("side_b/summary.json", summary)
        self.write("side_b/evidence.json", None)
        man = self.manifests["side_b"]
        man["evidence"]["model"] = [{"path": "evidence.json", "pointer": "/model"}]
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "not_covered", "null_artifact")

    def test_invalid_nonfinite_side_does_not_leak_into_report(self):
        man = self.manifests["side_b"]
        man["side"] = float("nan")
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "incomparable", "side_identity")

    def test_relative_paths_and_evidence_pointer_escapes(self):
        man = self.manifests["side_b"]
        self.write("external.json", {"a/b": {"~value": self.common["model"]}})
        man["evidence"]["model"] = [{"path": "../external.json", "pointer": "/a~1b/~0value"}]
        self.write("side_b/manifest.json", man)
        r = self.compare()
        self.assertEqual(r["status"], "no_thresholds")
        self.assertIn(str((self.root / "external.json").resolve()), r["evidence_paths"])

    def test_evidence_paths_resolve_directory_symlinks(self):
        self.write("external.json", {"model": self.common["model"]})
        alias = self.root / "evidence-alias"
        alias.symlink_to(self.root.resolve(), target_is_directory=True)
        man = self.manifests["side_b"]
        man["evidence"]["model"] = [{"path": "../evidence-alias/external.json", "pointer": "/model"}]
        self.write("side_b/manifest.json", man)
        report = self.compare()
        self.assertEqual(report["status"], "no_thresholds")
        canonical = str((self.root / "external.json").resolve())
        self.assertIn(canonical, report["evidence_paths"])
        self.assertEqual(report["evidence_paths"].count(canonical), 1)
        self.assertNotIn(str(alias / "external.json"), report["evidence_paths"])

    def test_v2_native_stacks_and_core_rule_ownership(self):
        man = self.manifests["side_b"]
        man["side_config"] = {"framework": "other", "kernel": "fused", "internal_dtype": None}
        self.write("side_b/manifest.json", man)
        report = self.compare()
        self.assertEqual(report["status"], "no_thresholds")
        self.assertEqual(report["core_version"], "3.0.0")
        self.assertEqual(man["actual"]["weight_dtype"], "bfloat16")
        self.assertEqual(man["actual"]["logprob_dtype"], "float32")
        self.assertTrue(CORE.CORE_RULES.keys().isdisjoint(man["actual"]))
        self.assertTrue(CORE.CORE_RULES.keys().isdisjoint(man["evidence"]))
        for key, rule in report["rules"].items():
            self.assertTrue(rule["declaration_valid"], key)
            self.assertTrue(rule["executed_at"], key)
        self.assertEqual(report["rules"]["percentile"]["executed_at"], ["metrics.p99"])

    def test_autocast_supported_policies(self):
        for dtype in (None, "float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                self.common["autocast"] = {"enabled": dtype is not None, "target_dtype": dtype,
                                           "scope": "model_forward"}
                self.write("contract.json", self.contract)
                self.make_side("side_a", [-1, -2, -3, -4])
                self.make_side("side_b", [-1, -2, -3, -4])
                self.assertEqual(self.compare()["status"], "no_thresholds")

    def test_autocast_exact_shape_and_types(self):
        valid = copy.deepcopy(self.common["autocast"])
        invalid = [False, "bfloat16", {}, {**valid, "extra": 1},
                   {k: v for k, v in valid.items() if k != "enabled"}]
        invalid += [{**valid, "enabled": x} for x in (0, 1, None, "false")]
        invalid += [{**valid, "target_dtype": x} for x in ("float32", False, 0)]
        invalid += [{**valid, "enabled": True, "target_dtype": x}
                    for x in (None, "float64", True, [], {})]
        invalid += [{**valid, "scope": x} for x in (None, "scoring", True)]
        for owner in ("contract", "side_b"):
            for value in invalid:
                with self.subTest(owner=owner, value=value):
                    doc = copy.deepcopy(self.contract if owner == "contract" else self.manifests[owner])
                    doc["common" if owner == "contract" else "actual"]["autocast"] = value
                    path = "contract.json" if owner == "contract" else "side_b/manifest.json"
                    self.write(path, doc)
                    self.assertBlocked(self.compare(), "incomparable", "condition_invalid")
                    self.write(path, self.contract if owner == "contract" else self.manifests[owner])

    def test_autocast_conflict_missing_and_object_evidence(self):
        man = self.manifests["side_b"]
        man["actual"]["autocast"] = {"enabled": True, "target_dtype": "bfloat16", "scope": "model_forward"}
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "incomparable", "condition_mismatch")
        man["actual"]["autocast"] = copy.deepcopy(self.common["autocast"])
        refs = man["evidence"].pop("autocast")
        self.write("side_b/manifest.json", man)
        report = self.compare()
        self.assertBlocked(report, "not_covered", "missing_evidence")
        self.assertEqual(report["rules"]["percentile"]["executed_at"], [])
        self.assertTrue(report["rules"]["reduction"]["executed_at"])
        man["evidence"]["autocast"] = refs
        self.write("side_b/manifest.json", man)
        for value in ({"enabled": 0, "target_dtype": None, "scope": "model_forward"},
                      {**self.common["autocast"], "extra": True}, False):
            with self.subTest(evidence=value):
                env = self.read("side_b/env0.json")
                env["observed"]["autocast"] = value
                self.write("side_b/env0.json", env)
                self.assertBlocked(self.compare(), "incomparable", "evidence_mismatch")

    def test_removed_and_extra_keys_rejected_at_exact_location(self):
        for container in ("common", "actual", "evidence"):
            keys = ("compute_dtype", "extra") if container == "common" else (
                "compute_dtype", "extra", "target_shift", "reduction", "percentile")
            for key in keys:
                with self.subTest(key=key):
                    doc = copy.deepcopy(self.contract if container == "common" else self.manifests["side_b"])
                    doc[container][key] = "old"
                    path = "contract.json" if container == "common" else "side_b/manifest.json"
                    self.write(path, doc)
                    report = self.compare()
                    code = "unknown_evidence_field" if container == "evidence" else "unknown_common_field"
                    self.assertBlocked(report, "incomparable", code)
                    owner = "contract" if container == "common" else "side_b"
                    self.assertTrue(any(b["where"] == f"{owner}.{container}.{key}" for b in report["blockers"]))
                    self.write(path, self.contract if container == "common" else self.manifests["side_b"])

    def test_rejected_core_declarations_and_shift_arrays(self):
        for key, value in (("target_shift", 0), ("reduction", "sample_mean"), ("percentile", "rank_mean")):
            with self.subTest(key=key):
                doc = copy.deepcopy(self.contract)
                doc["common"][key] = value
                self.write("contract.json", doc)
                report = self.compare()
                self.assertBlocked(report, "incomparable", "condition_invalid")
                self.assertFalse(report["rules"][key]["declaration_valid"])
                self.assertEqual(report["rules"][key]["definition"], CORE.CORE_RULES[key])
                self.assertEqual(report["rules"]["percentile"]["executed_at"], [])
        self.write("contract.json", self.contract)
        # All three saved inputs agree, but the actual -1 relationship is wrong.
        data = copy.deepcopy(self.inputs)
        data["samples"][0]["prediction_positions"][0] += 1
        for name in ("inputs.json", "side_a/inputs.json", "side_b/inputs.json"):
            self.write(name, data)
        self.assertBlocked(self.compare(), "incomparable", "input_alignment")

    def test_target_scope_still_requires_actual_evidence(self):
        man = self.manifests["side_b"]
        del man["evidence"]["target_scope"]
        self.write("side_b/manifest.json", man)
        self.assertBlocked(self.compare(), "not_covered", "missing_evidence")

    def test_cli_v2_missing_evidence_thresholds_and_v1(self):
        command = [sys.executable, "-B", "-I", str(SCRIPT),
                   "--contract", str(self.root / "contract.json"),
                   "--side-a", str(self.root / "side_a/manifest.json"),
                   "--side-b", str(self.root / "side_b/manifest.json")]
        for mode in ("valid", "missing_evidence", "thresholds", "v1"):
            with self.subTest(mode=mode):
                contract = copy.deepcopy(self.contract)
                man = copy.deepcopy(self.manifests["side_b"])
                if mode == "missing_evidence":
                    del man["evidence"]["autocast"]
                elif mode == "thresholds":
                    contract["thresholds"] = {}
                elif mode == "v1":
                    contract["schema_version"] = 1
                self.write("contract.json", contract)
                self.write("side_b/manifest.json", man)
                output = self.root / (mode + ".json")
                result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if mode == "valid" else 1, result.stderr)
                report = self.read(output.name)
                self.assertEqual(report["schema_version"], 2)
                self.assertEqual(report["core_version"], "3.0.0")
                self.assertEqual(bool(report["rules"]["percentile"]["executed_at"]), mode == "valid")

    def test_cli_exclusive_create_and_blocked_output(self):
        output = self.root / "report.json"
        command = [sys.executable, "-B", "-I", str(SCRIPT),
                   "--contract", str(self.root / "contract.json"),
                   "--side-a", str(self.root / "side_a/manifest.json"),
                   "--side-b", str(self.root / "side_b/manifest.json"), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read("report.json")["status"], "no_thresholds")
        original = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), original)
        output.unlink()
        (self.root / "side_b/scores.jsonl").write_text("bad JSON")
        result = subprocess.run(command, capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.read("report.json")["status"], "execution_failed")


if __name__ == "__main__":
    unittest.main()
