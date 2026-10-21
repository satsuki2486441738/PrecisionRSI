"""Chart domains preserve signed measurements and distinguish observation coverage."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for chart model tests')
class EvaluationChartTests(unittest.TestCase):
    def evaluate(self, expression):
        script = ("const {chartModel,barGeometry,coverageModel} = require('./site/evaluations.js');"
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

    def test_error_scales_stay_within_comparison_unit_and_semantics(self):
        output = self.evaluate("""(() => {
          const metrics=[{id:'target-mae',unit:'nat',value:0.02},
            {id:'target-p99',unit:'nat',value:0.08},
            {id:'target-max',unit:'nat',value:0.1},
            {id:'gradients-rmse',unit:'gradient',value:999},
            {id:'tail-merged-js-mean',unit:'nat',value:0.5},
            {id:'side-norm-abs-diff',unit:'nat',value:20,side_values:{a:80,b:100}}];
          const c={side_ids:['a','b'],metrics};
          return {error:chartModel(metrics[0],c), js:chartModel(metrics[4],c),
            other:chartModel({...metrics[0],value:2},{...c,metrics:[{...metrics[0],value:2}]})};
        })()""")
        self.assertEqual(output['error']['kind'], 'error')
        self.assertEqual(output['error']['domain'], [0, 0.1])
        self.assertEqual(output['js']['kind'], 'divergence')
        self.assertEqual(output['js']['domain'], [0, 0.5])
        self.assertEqual(output['other']['domain'], [0, 2])

    def test_mathematical_domains_and_unknown_metrics(self):
        output = self.evaluate("""(() => {
          const metrics=[{id:'gradients-cosine',unit:'1',value:0.999},
            {id:'top1-agreement',unit:'fraction',value:1},
            {id:'top100-overlap',unit:'1',value:0},
            {id:'gradients-cosine',unit:'1',value:1.1},
            {id:'runtime-seconds',unit:'s',value:200},
            {id:'top1-agreement',unit:'percent',value:50},
            {id:'gradients-cosine',unit:'percent',value:0.5}];
          return metrics.map(m=>chartModel(m,{side_ids:['a','b'],metrics}));
        })()""")
        self.assertEqual(output[0]['domain'], [-1, 1])
        self.assertEqual(output[1]['domain'], [0, 1])
        self.assertEqual(output[2]['domain'], [0, 1])
        self.assertEqual(output[2]['rows'][0]['geometry']['width'], 0)
        self.assertEqual(output[3]['kind'], 'none')
        self.assertEqual(output[4]['kind'], 'none')
        self.assertEqual(output[5]['kind'], 'none')
        self.assertEqual(output[6]['kind'], 'none')

    def test_all_zero_errors_and_coverage_are_not_acceptance(self):
        output = self.evaluate("""(() => {
          const m={id:'target-mae',unit:'nat',value:0};
          return {chart:chartModel(m,{side_ids:['a','b'],metrics:[m]}),
            partial:coverageModel({observed:3,expected:4}),
            zero:coverageModel({observed:0,expected:4}),
            over:coverageModel({observed:5,expected:4}),
            empty:coverageModel({observed:0,expected:0})};
        })()""")
        self.assertEqual(output['chart']['domain'], [0, 0])
        self.assertEqual(output['chart']['rows'][0]['geometry']['width'], 0)
        self.assertEqual(output['partial']['ratio'], 0.75)
        self.assertEqual(output['partial']['width'], 75)
        self.assertEqual(output['zero']['width'], 0)
        self.assertEqual(output['over']['ratio'], 1.25)
        self.assertEqual(output['over']['width'], 100)
        self.assertFalse(output['empty']['drawable'])


if __name__ == '__main__':
    unittest.main()
