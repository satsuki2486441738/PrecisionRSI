"""Display names retain experiment boundaries without rewriting history records."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for display title tests')
class EvaluationPresentationTests(unittest.TestCase):
    def evaluate(self, expression):
        script = ("const {titleFor}=require('./site/evaluation-presentation.js');"
                  "const records=require('./site/data/evaluations.json').evaluations;"
                  "const byId=(id)=>records.find(r=>r.id===id);"
                  "console.log(JSON.stringify(" + expression + ")); ")
        result = subprocess.run(['node', '-e', script], cwd=ROOT, text=True,
                                encoding='utf-8', capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_every_actual_record_has_a_concise_nonmutating_name(self):
        result = self.evaluate("""(() => {
          const before=JSON.stringify(records);
          const names=records.map(r=>({id:r.id,title:titleFor(r,'模型标签')}));
          return {names,unchanged:JSON.stringify(records)===before};
        })()""")
        self.assertGreaterEqual(len(result['names']), 71)
        self.assertTrue(result['unchanged'])
        for row in result['names']:
            with self.subTest(record=row['id']):
                self.assertTrue(row['title'].endswith(' · 模型标签'))
                self.assertEqual(row['title'].count(' · '), 2)
                self.assertLessEqual(len(row['title']), 65)
                self.assertNotRegex(row['title'], r'20\d{6}|seed|种子|原生|已完成|精度评测|9382|RTX|H100|H200')

    def test_distinct_inference_execution_modes_survive_shortening(self):
        names = self.evaluate("""[
          'llamacpp-luna-smollm2-135m-20261019',
          'llamacpp-local-smollm2-135m-instruct-20261019',
          'llama-cuda-smollm2-135m-instruct-20261021',
          'lmdeploy-local-smollm2-135m-instruct-20261019',
          'lmdeploy-cuda-smollm2-135m-instruct-20261021'
        ].map(id=>titleFor(byId(id),'SmolLM2-135M-Instruct'))""")
        self.assertEqual(len(set(names)), len(names))
        self.assertIn('CPU 冷上下文推理', names[0])
        self.assertIn('CPU 缓存推理', names[1])
        self.assertIn('CUDA 缓存推理', names[2])
        self.assertIn('CUDA 双引擎重复预填充', names[3])
        self.assertIn('CUDA 缓存推理', names[4])

    def test_precision_and_native_attention_routes_remain_visible(self):
        precision, cann, trt = self.evaluate("""[
          titleFor(byId('torch-bf16-fp16-20261021'),'Qwen3.5-9B-Base'),
          titleFor(byId('llamacpp-cann-qwen3-0-6b-20261020'),'Qwen3-0.6B'),
          titleFor(byId('tensorrt-llm-seeded-llama-20261020'),'tiny Llama')
        ]""")
        for token in ('PyTorch', 'BF16 ↔ FP16', '完整前向', '缓存推理'):
            self.assertIn(token, precision)
        self.assertIn('CPU ↔ CANN 缓存推理', cann)
        self.assertIn('CUDA TRTLLM 缓存推理', trt)

    def test_sft_checkpoint_ddp_component_and_training_kind_are_distinct(self):
        names = self.evaluate("""[
          'trl-sfttrainer-checkpoint-20261021',
          'megatron-bridge-pretrained-sft-npu-20261020',
          'megatron-bridge-qwen3-06b-h100-native-20261022',
          'trl-qwen3-06b-h100-ddp-native-20261022',
          'trl-tiny-qwen2-h100-ddp-native-20261022',
          'trl-tiny-qwen2-sft-local-20261019',
          'trl-qwen3-06b-cuda-native-20261021t175700z'
        ].map(id=>titleFor(byId(id),'模型'))""")
        self.assertIn('LoRA checkpoint 恢复', names[0])
        self.assertIn('Ascend fork checkpoint 冷恢复 SFT', names[1])
        self.assertIn('CUDA checkpoint 冷恢复 SFT', names[2])
        self.assertIn('4卡 DDP LoRA SFT', names[3])
        self.assertIn('4卡 DDP 全参 SFT', names[4])
        self.assertIn('CUDA SFT 组件', names[5])
        self.assertIn('CUDA LoRA SFT', names[6])

    def test_rl_worker_component_untied_and_full_workflows_remain_visible(self):
        names = self.evaluate("""[
          'nemo-rl-tiny-qwen2-worker-20261020',
          'nemo-rl-qwen3-06b-official-cli-20261020',
          'nemo-pretrained-untied-fp32-h100-20261022',
          'openrlhf-rl-local-smollm2-135m-20261019',
          'ms-swift-rl-local-smollm2-135m-20261019',
          'openrlhf-pretrained-cuda-h100-20261022',
          'roll-pretrained-cuda-h100-20261022'
        ].map(id=>titleFor(byId(id),'模型'))""")
        self.assertIn('Ascend PPO worker', names[0])
        self.assertIn('Ascend 奖励驱动 GRPO', names[1])
        self.assertIn('CUDA untied GRPO', names[2])
        self.assertIn('CUDA PPO 组件', names[3])
        self.assertIn('CPU GRPO 组件', names[4])
        self.assertIn('CUDA 两卡 PPO', names[5])
        self.assertIn('CUDA 两卡 GRPO', names[6])

    def test_future_metadata_and_unknown_route_fallback_do_not_need_id_tables(self):
        names = self.evaluate("""[
          titleFor({id:'new-case',framework:'FutureEngine',task:'inference',
            title:'FutureEngine · parameterized full-forward inference · FutureModel · 2026-11-01',
            model:{name:'FutureModel'},sides:[]},'FutureModel'),
          titleFor({id:'unseen-trl-case',title:'TRL kernel SFT',task:'sft',
            sides:[{hardware:'NVIDIA future GPU',settings:{training:'LoRA',
              topology:{world_size:8},execution:'native DDP'}}]},'FutureModel'),
          titleFor({framework:{name:'AnotherEngine'},task:'inference',title:'',
            sides:[{hardware:'CPU',settings:{scope:'cached fixed-history scoring'}}]},'模型'),
          titleFor({},'模型')
        ]""")
        self.assertEqual(names[0], 'FutureEngine · 参数化 完整前向推理 · FutureModel')
        self.assertEqual(names[1], 'TRL · kernel 8卡 DDP LoRA SFT · FutureModel')
        self.assertEqual(names[2], 'AnotherEngine · CPU 缓存推理 · 模型')
        self.assertEqual(names[3], '评测 · 实验 · 模型')

    def test_cross_library_benchmark_names_keep_libraries_and_phase_once(self):
        names = self.evaluate("""[
          'skills-native-sglangtf-initial-20261024',
          'skills-native-sglangtf-extra-20261024'
        ].map(id=>titleFor(byId(id),'Qwen3.5-9B-Base'))""")
        self.assertNotEqual(names[0], names[1])
        for name in names:
            self.assertLessEqual(len(name), 65)
            for label in ('Skills A/B', 'SGLang', 'Transformers'):
                self.assertEqual(name.count(label), 1)
            self.assertTrue(name.endswith(' · Qwen3.5-9B-Base'))
        self.assertIn('首次', names[0])
        self.assertIn('合并复用', names[1])

    def test_browser_global_and_commonjs_expose_the_same_api(self):
        result = self.evaluate("""(() => {
          const fs=require('fs'),vm=require('vm'),context={};
          vm.runInNewContext(fs.readFileSync('./site/evaluation-presentation.js','utf8'),context);
          return context.EvaluationPresentation.titleFor(byId('trl-sfttrainer-lora-20261021'),'模型');
        })()""")
        self.assertEqual(result, 'TRL · CUDA ↔ Ascend LoRA SFT · 模型')


if __name__ == '__main__':
    unittest.main()
