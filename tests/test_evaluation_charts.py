"""Only meaningful two-side differences need bars; summaries remain precise text."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for chart model tests')
class EvaluationChartTests(unittest.TestCase):
    def evaluate(self, expression):
        script = ("const {chartModel,barGeometry} = require('./site/evaluations.js');"
                  "console.log(JSON.stringify(" + expression + ")); ")
        result = subprocess.run(['node', '-e', script], cwd=ROOT, text=True,
                                capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_negative_and_mixed_sign_bars_keep_a_common_zero(self):
        rows = self.evaluate('[barGeometry(-2,[-4,0]),barGeometry(-2,[-4,4]),'
                             'barGeometry(2,[-4,4]),barGeometry(0,[0,0])]')
        self.assertEqual(rows[0], {'left': 50, 'width': 50, 'zero': 100})
        self.assertEqual(rows[1], {'left': 25, 'width': 25, 'zero': 50})
        self.assertEqual(rows[2], {'left': 50, 'width': 25, 'zero': 50})
        self.assertEqual(rows[3]['width'], 0)

    def test_extreme_finite_and_tiny_values_do_not_invent_minimum_width(self):
        rows = self.evaluate('[barGeometry(-1e308,[-1e308,1e308]),'
                             'barGeometry(1e308,[-1e308,1e308]),'
                             'barGeometry(1e-20,[0,1]),barGeometry(0,[0,1])]')
        self.assertEqual(rows[0], {'left': 0, 'width': 50, 'zero': 50})
        self.assertEqual(rows[1], {'left': 50, 'width': 50, 'zero': 50})
        self.assertAlmostEqual(rows[2]['width'], 1e-18, delta=1e-30)
        self.assertEqual(rows[3]['width'], 0)
        self.assertIsNone(self.evaluate('barGeometry(NaN,[0,1])'))
        self.assertIsNone(self.evaluate('barGeometry(1,[0,Infinity])'))

    def test_recorded_sides_keep_order_sign_and_original_values(self):
        output = self.evaluate("""(() => {
          const m={id:'signed-loss-abs-diff',unit:'loss',value:0.25,
            side_values:{native:-0.5,cpu:-0.75}};
          const c={side_ids:['cpu','native'],metrics:[m]};
          const before=JSON.stringify(c); const chart=chartModel(m,c);
          return {chart,unchanged:JSON.stringify(c)===before};
        })()""")
        chart = output['chart']
        self.assertTrue(output['unchanged'])
        self.assertEqual(chart['kind'], 'sides')
        self.assertEqual(chart['domain'], [-0.75, 0])
        self.assertEqual([(r['id'], r['value']) for r in chart['rows']],
                         [('cpu', -0.75), ('native', -0.5)])

    def test_single_summary_metrics_never_get_decorative_bars(self):
        output = self.evaluate("""(() => {
          const metrics=[{id:'target-mae',unit:'nat',value:0.02},
            {id:'gradients-cosine',unit:'1',value:0.999},
            {id:'top1-agreement',unit:'fraction',value:1},
            {id:'tail-merged-js-mean',unit:'nat',value:0.5},
            {id:'exact-arrays',unit:'arrays',value:298},
            {id:'target-max',unit:'nat',value:0}];
          return metrics.map(m=>chartModel(m,{side_ids:['a','b'],metrics}));
        })()""")
        for chart in output:
            self.assertEqual(chart['kind'], 'none')
            self.assertIsNone(chart['domain'])
            self.assertEqual(chart['reason'], 'summary')

    def test_equal_sides_retain_the_pair_without_identical_bars(self):
        output = self.evaluate("""[0,1,-1].map(v=>chartModel(
          {value:0,side_values:{a:v,b:v}}, {side_ids:['a','b']}))""")
        for number, chart in zip((0, 1, -1), output):
            self.assertEqual(chart['kind'], 'none')
            self.assertEqual(chart['reason'], 'equal-sides')
            self.assertEqual(chart['rows'], [{'id': 'a', 'value': number},
                                             {'id': 'b', 'value': number}])

    def test_small_real_differences_have_no_arbitrary_display_cutoff(self):
        chart = self.evaluate("chartModel({value:1e-15,side_values:{a:1,b:1+1e-15}}, {side_ids:['a','b']})")
        self.assertEqual(chart['kind'], 'sides')
        self.assertNotEqual(chart['rows'][0]['value'], chart['rows'][1]['value'])
        invalid = self.evaluate("chartModel({value:1,side_values:{a:1}}, {side_ids:['a','missing']})")
        self.assertEqual(invalid['kind'], 'none')
        self.assertEqual(invalid['reason'], 'invalid-sides')


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for settings formatter tests')
class EvaluationSettingsTests(unittest.TestCase):
    def evaluate(self, expression):
        script = ("const {settingLabel,formatSettingValue} = require('./site/evaluation-settings.js');"
                  "console.log(JSON.stringify(" + expression + ")); ")
        result = subprocess.run(['node', '-e', script], cwd=ROOT, text=True,
                                capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_forward_shape_is_readable_without_changing_execution_meaning(self):
        value = self.evaluate("formatSettingValue('one6-token prefill plus two single-token cached decode calls','forward_shape')")
        for part in ('1', '6', '2', '预填充', '缓存解码'):
            self.assertIn(part, value)
        self.assertNotIn('one6', value)

    def test_empty_false_zero_and_unknown_values_are_distinct(self):
        values = self.evaluate("[0,false,true,null,''].map(v=>formatSettingValue(v,'custom_field'))")
        self.assertEqual(values[0], '0')
        self.assertEqual(len(set(values)), 5)
        self.assertEqual(self.evaluate("formatSettingValue('unrecorded custom string','custom_field')"),
                         'unrecorded custom string')

    def test_familiar_setting_keys_have_human_labels(self):
        labels = self.evaluate("['model','inputs','numerics','execution','forward_shape'].map(settingLabel)")
        for original, label in zip(('model', 'inputs', 'numerics', 'execution', 'forward_shape'), labels):
            self.assertNotEqual(label, original)
            self.assertNotIn('_', label)
        self.assertTrue(self.evaluate("settingLabel('custom_future_field')"))

    def test_formatter_keeps_supplied_markup_literal(self):
        value = '<img src=x onerror="alert(1)"> & literal'
        self.assertEqual(self.evaluate('formatSettingValue(' + json.dumps(value) + ",'custom_field')"), value)


if __name__ == '__main__':
    unittest.main()
