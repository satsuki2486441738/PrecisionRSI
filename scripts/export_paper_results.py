#!/usr/bin/env python3
"""Export allowlisted paper aggregates (not raw traces) into a crawlable writing bundle.
Usage: python -B -m scripts export-paper --output NEW_DIRECTORY
Outputs sources/, reports/, figures/, results.json, values.csv, manifest.json,
verify.py, README.md and paper-writing-results.zip. Existing output is rejected.
JSON pointers address sanitized exports; source and export hashes are distinct.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'projects/torch-gpu-npu/cases'
FOLLOW = CASES / 'qwen35-qwen38-followup/runs/compare-qwen-followup-scoped-final-delivery-20261107T225131Z/package'
OPERATOR = CASES / 'paper-operator-version-followup/runs/compare-paper-final-publication-figures-delivery-20261108T045510Z/package'
PRIVATE = re.compile(r'/(?:Users|tmp|home|root|private|workspace|mnt)/[^\s\"<>`)]+|\bmodelmate[\w.-]*\b|\b(?:\d{1,3}\.){3}\d{1,3}\b', re.I)
OMIT = {'physical_devices', 'physical_device', 'logical_device', 'preload_resource_observations_by_arm_rank', 'allocation', 'effective_device_environment'}
SUPPLEMENTS = (
    'deepseek41-upstream-extension-20261108',
    'deepseek41-compressor-2k-20261108',
    'deepseek41-fresh-process-20261108',
    'deepseek41-native-indexer-20261108',
    'deepseek41-native-query-tail-20261108',
    'deepseek41-adapted-core-20261109',
    'deepseek41-full40-attention-20261109',
    'deepseek41-paired-output-head95-20261109',
    'deepseek41-http128-20261109',
    'deepseek41-native-producer-key-cache-20261109',
    'deepseek41-native-query-key-selection-20261109',
    'deepseek41-retained-mode2-20261109',
    'deepseek41-six-forward-selectors-20261109',
)
NOTES = [
    'A writing snapshot of existing aggregates, not a new experiment or an acceptance decision. Null means unspecified; missing coverage is not a pass.',
    'Use source configurations and report scope with every citation. GPU is not a mathematical oracle. DeepSeek mixed-FP8/FP4 TP4 versus adapted-BF16 TP8/EP8/offload48 is not a hardware-only comparison.',
    'Qwen3.5 default NPU checkpoint continuation was 298/448 exact arrays; native deterministic level 1 reached 448/448. This supports a configuration-scoped conclusion.',
    'Independent examples, dependent coordinates, repeated processes and overlapping operator prefixes are different units. Do not pool them as independent samples.',
    'The numeric ledger includes configuration/count fields and dependent observations as well as measurements. Unspecified units must be read in the source/report, never guessed.',
    'Operator alternate stacks change together. Active-cotangent diagnostics are post-outcome, not untouched validation. H200 is user-confirmed; raw H20Z metadata is retained.',
    'DeepSeek original ledger remains 191/275. A separately reviewed adapted configuration reaches 275/275, with 191 guarded reused observations and 84 new. All 3168 targets are reused, not new quality experiments. Service/training gaps remain explicit; whole-task completion is false.',
    'DeepSeek paired output-head training retains 95 dependent same-coordinate calls per side, 83 updates and 12 reserved evaluations. Both runtimes independently pass continuation/repeat replay. Initial A is near-equal but not byte-identical; later adapter states follow separate trajectories. No backbone autograd or hardware-only causal claim.',
    'DeepSeek adapted HTTP128 completed 128 requests, including nine actual 32K histories, 3168 selected targets and 24 candidate terms across 18 sequences. Ordinary 78 full-prefill rows and 32 fresh-prefill versus cached-core teacher rows are different execution groups. GPU service inventories remain separate pending work.',
    'DeepSeek sources labeled HISTORICAL 2026-11-06 preserve that evidence boundary. Their old gap statements, including then-untested fresh-process or HTTP scopes, are superseded by the current reviewed report and separately scoped newer supplements. Historical measurements and missing original semantics remain unchanged.',
    'The outer writing verifier checks byte integrity and exact source pointers. Most supplements contain aggregates only. The HTTP128 supplement additionally includes 128 original generated full-vocabulary arrays and its own numerical verifier; its selected prompt-scalar/raw-response proof remains bound to retained native independent evidence. Read each supplement scope before claiming replay.',
    'The native producer-to-key/cache supplement retains four dependent serialized capsules. Its private full projection review covers 6684672 key coordinates; its portable projection replay samples 256 rows (32768 outputs), while downstream normalization/RoPE/quantization and 60709632 reconstructed cache bytes are replayed in full. Exported FP32-scale reconstruction and actual FP16-cache dequantization are separate metrics. This does not cover a same-live-tensor chain, query/score/top-k, full-model cache semantics or backbone training.',
    'The separately reviewed retained native query/key QLI supplement composes four serialized tail4/all32-head capsules, preserving all20 source operands and74 actual numeric/raw pairs. It does not rewrite the earlier producer-key supplement scope. ExactINT64 dot then actualFP16 scales/weights in FP64 yields descriptive top512 overlaps11x512/4x511/1x510; native own full scores remain unavailable. L2 mode3ordinary/L20 mode1candidate source do not establish mode2consumer equivalence. 2K all256visible candidates are trivial; 32K top2048 overlaps2048/2047/2047/2047. No same-live-tensor, fullmodel, TP, training, service or numerical-acceptance claim.',
    'The retained same-L20 mode2 supplement is a bounded component test using actual L20 retained candidates, 20 source operands, 21 numeric/raw pairs and 131072 derived score coordinates. It does not establish the original L20-to-L24/28/32/36 crosslayer chain, native own scores, attention, full-model, TP, training, hardware cause or acceptance. Its portable ZIP includes bounded arrays and numerical replay.',
    'The six-forward selector supplement summarizes independently reviewed original rank0 tail4 arithmetic: 30 selector cases, 120 dependent selected-query cases and 1740800 derived score coordinates. Four all-visible candidate cohorts and two selective32768-key cohorts remain separate; the first16384-key chunk has no negative padding but is all-visible. The single32768-row forward uses an adapted full40 configuration, not original-default unchunked32K closure. Its sanitized ZIP contains aggregates only and verifies bytes/CSV, not independent numerical replay; native own scores, full-history/allTP coverage, hardware cause and acceptance remain unavailable.',
]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()

def sanitize(value):
    if isinstance(value, dict):
        return {(('private-source-' + sha(k.encode())[:16]) if k.startswith(('projects/', '/')) else k): sanitize(v) for k, v in value.items() if k not in OMIT and 'uuid' not in k.lower()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return PRIVATE.sub('[private reference removed]', value).replace('q35-cache-[private reference removed]', 'Q35 secondary-cache successor')
    return value

def pointer_escape(key):
    return str(key).replace('~', '~0').replace('/', '~1')

def unit_for(key, parent):
    if isinstance(parent, dict) and isinstance(parent.get('unit'), str) and key in {'value','before','after'}:
        return parent['unit']
    if key.endswith('_nat_per_token'): return 'nat/token'
    if key.endswith('_nat') or key in {'scores_nat','mae_nat','p99_nat'}: return 'nat'
    if key.endswith('_seconds') or key == 'seconds': return 's'
    if key.endswith('_bytes'): return 'byte'
    if key == 'tokens_per_second': return 'token/s'
    return 'unspecified; see source context'

def numbers(value, pointer='', inherited_unit=None):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from numbers(v, pointer + '/' + pointer_escape(k), unit_for(k, value))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from numbers(v, pointer + '/' + str(i), inherited_unit)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield pointer, value, inherited_unit or 'unspecified; see source context'

def registry():
    evidence = ROOT / 'community/site/evidence'
    specs = [
        ('original-repairs','Original manuscript / repair history',ROOT/'community/site/data/repairs.json'),
        ('model-evaluations','Qwen / DeepSeek model evaluations',ROOT/'community/site/data/evaluations.json'),
        ('qwen-followup','Qwen follow-up / final',FOLLOW/'results.json'),
        ('qwen-diagnostics','Qwen follow-up / diagnostic and held-out cohorts',FOLLOW/'historical/interim/results.json'),
        ('operator-original','Operator / original conditions across stacks',OPERATOR/'results.json'),
        ('operator-active','Operator / post-outcome active cotangents',OPERATOR/'active-results.json'),
    ]
    for name in ['mindspeed-gemm-policy','mindspeed-rmsnorm-layerwise','mindspeed-rmsnorm-training','mindspeed-selective-projection','mindspeed-decoder-trace','cross-platform-rmsnorm-policy-20261025','cross-platform-sft-causal-continuation-20261025','qwen38-precision-20261106','deepseek41-precision-20261106']:
        specs.append((name,name,evidence/(name+'.json')))
    for name in ['core-measurements','gpu-performance','npu-performance','head-summary','gap-ledger','runtime-facts','framework-facts','terminal-cleanup-summary']:
        specs.append(('deepseek-'+name,'DeepSeek / HISTORICAL 2026-11-06 / '+name,evidence/'deepseek41-precision-20261106'/(name+'.json')))
    for name in SUPPLEMENTS:
        specs.append((name, 'DeepSeek / reviewed supplement / '+name, ROOT/'community/site/data'/(name+'.json')))
    return specs

def filtered_history(identity, data):
    if identity == 'model-evaluations':
        data['evaluations'] = [x for x in data['evaluations'] if not x['id'].startswith(('skills-','rsi-')) and (x['id'].startswith('deepseek41-') or any(t in x['id'] for t in ['qwen38-native','trl-sfttrainer','torch-bf16-fp16','autonomous-cross-platform','cross-platform-sft-causal']))]
    if identity == 'original-repairs':
        data['repairs'] = [x for x in data['repairs'] if x['id'].startswith(('mindspeed-','cross-platform-rmsnorm'))]
    return data

def highlights(sources):
    by_id = {s['id']: s['data'] for s in sources}
    specs = [
        ('Qwen3.5 NPU checkpoint / default', 'qwen-diagnostics', '/Qwen35/historical_TRL/comparisons/default/continuous-restored/exact_equal_arrays', 'exact arrays / 448', 'Continuous vs restored, default policy; configuration-scoped.'),
        ('Qwen3.5 NPU checkpoint / level 1', 'qwen-diagnostics', '/Qwen35/historical_TRL/comparisons/native-level1/continuous-restored/exact_equal_arrays', 'exact arrays / 448', 'Native deterministic level 1; same workload. Do not claim inherent NPU non-determinism.'),
        ('Qwen3.5 held-out / distribution', 'qwen-diagnostics', '/Qwen35/heldout/max_shared_full_vocabulary_JS_nat', 'nat', 'Maximum JS; 9 independent examples, 92 causal rows per side.'),
        ('Qwen3.8 held-out / distribution', 'qwen-diagnostics', '/Qwen38/shared_full_vocab_JS_max_nat', 'nat', 'Maximum JS; 9 independent examples, 36 full-forward pairs. GPU placement differs from NPU TP8.'),
        ('Qwen3.5 secondary cache / distribution', 'qwen-followup', '/secondary_cache/cross_peer_distributions/maximum_JS_nat', 'nat', '36 arrays / 144 rows; secondary-cache cohort does not replace canonical primary cache.'),
        ('DeepSeek target log-probability MAE', 'deepseek-core-measurements', '/targets/mae_nat', 'nat', '3168 paired target coordinates; original mixed GPU vs BF16-adapted NPU.'),
        ('DeepSeek target log-probability p99', 'deepseek-core-measurements', '/targets/p99_nat', 'nat', 'Global linear p99 over paired target coordinates; configuration-confounded.'),
        ('DeepSeek target log-probability maximum', 'deepseek-core-measurements', '/targets/max_abs_nat', 'nat', 'Tail coordinate retained in source; do not infer tails from mean NLL.'),
        ('DeepSeek absolute mean NLL gap', 'deepseek-core-measurements', '/targets/nll_abs_diff_nat_per_token', 'nat/token', 'Absolute difference of side mean NLLs, distinct from mean absolute target log-probability error.'),
        ('GPU GEMM cross-stack agreement', 'operator-original', '/versions/gpu/exact_observations', 'exact observations / 192', 'Same captured inputs and weights; dependent outputs, not 192 independent model samples.'),
        ('NPU original operator cross-stack agreement', 'operator-original', '/versions/npu/exact_observations', 'exact observations / 660', '649 exact; remaining dgamma differences coexist with within-stack variability.'),
        ('NPU active-cotangent cross-stack agreement', 'operator-active', '/versions/npu/exact_observations', 'exact observations / 540', 'Post-outcome active diagnostic cohort; do not treat as untouched validation.'),
        ('DeepSeek separately adapted core coverage', 'deepseek41-adapted-core-20261109', '/coverage/adapted_paired', 'paired cells / 275', 'Original unadapted ledger remains 191/275; coverage is not acceptance.'),
        ('DeepSeek adapted mixed-tensor RMSE', 'deepseek41-adapted-core-20261109', '/tensors/metrics/rmse/value', 'mixed tensor units', 'Element-weighted logits and mHC diagnostic over actual 550 arrays; not accuracy.'),
        ('DeepSeek full40 attention tail residual', 'deepseek41-full40-attention-20261109', '/summary/maximum_absolute_residual', 'attention units', '480 rank-zero layer-forwards and 1920 saved tail-query rows; FP64 does not emulate native rounding.'),
        ('DeepSeek GPU fresh frozen-backbone calls', 'deepseek41-adapted-core-20261109', '/gpu_online_head/actual_complete_backbone_forward_calls', 'calls across 4 ranks', 'Output-head-only LoRA; dependent calls, not backbone autograd or independent tasks.'),
        ('DeepSeek NPU fresh frozen-backbone calls', 'deepseek41-paired-output-head95-20261109', '/native_forward_calls/npu', 'calls across 8 ranks', '95 per rank; output-head LoRA only, not backbone autograd or independent tasks.'),
        ('DeepSeek paired output-head coordinates', 'deepseek41-paired-output-head95-20261109', '/paired_calls', 'dependent paired calls', '83 update calls and 12 reserved evaluations; runtime-specific trajectories from near-equal initial A.'),
    ]
    rows = []
    for title, identity, pointer, unit, scope in specs:
        value = by_id[identity]
        for part in pointer.split('/')[1:]:
            value = value[part]
        rows.append(dict(title=title, source_id=identity, pointer=pointer, value=value, unit=unit, scope=scope))
    return rows

def export(output):
    output.mkdir(parents=True, exist_ok=False)
    sources, ledger, assets = [], [], []
    def write(name, data):
        p = output/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for identity, title, path in registry():
        raw = path.read_bytes()
        data = sanitize(filtered_history(identity, json.loads(raw)))
        payload = encoded(data)
        name = 'sources/'+identity+'.json'
        write(name,payload)
        sources.append(dict(id=identity,title=title,file=name,source_sha256=sha(raw),export_sha256=sha(payload),data=data))
        for pointer, value, unit in numbers(data):
            ledger.append(dict(source_id=identity,pointer=pointer,value=value,unit=unit,kind='numeric field; interpret with source configuration and scope'))
    docs = [('qwen35',ROOT/'reports/qwen35-precision-report.md'),('qwen38',ROOT/'reports/qwen38-precision-report.md'),('deepseek41',ROOT/'reports/deepseek41-precision-report.md'),('qwen-followup',FOLLOW/'report.md'),('qwen-diagnostics',FOLLOW/'historical/interim/.github/qwen35-qwen38-followup-interim-report.md'),('operator',OPERATOR/'report.md')]
    figures = [ROOT/'reports/qwen35-precision-trajectory.svg',ROOT/'reports/qwen38-precision-trajectory.svg',ROOT/'reports/deepseek41-precision-diagnostics.svg']
    figures += sorted((FOLLOW/'figures').glob('*.svg')) + sorted((FOLLOW/'historical/interim/.github/figures').glob('*.svg')) + sorted((OPERATOR/'figures').glob('*.svg'))
    asset_map = {path.name: '../figures/'+str(i+1)+'-'+path.name for i, path in enumerate(figures)}
    asset_map.update({path.name: '../sources/'+identity+'.json' for identity, _, path in registry() if path.name != 'results.json'})
    asset_map.update({path.name: identity+'.md' for identity, path in docs})
    def report_links(text, identity):
        def replace(match):
            image, label, target = match.groups()
            if re.match(r'[a-z]+:|#', target):
                return match.group(0)
            basename = target.split('#')[0].split('/')[-1]
            if basename == 'results.json':
                name = {'qwen-followup':'qwen-followup','qwen-diagnostics':'qwen-diagnostics','operator':'operator-original'}.get(identity)
                destination = '../sources/'+name+'.json' if name else None
            elif basename == 'active-results.json':
                destination = '../sources/operator-active.json'
            else:
                destination = asset_map.get(basename)
            if destination:
                return image+'['+label+']('+destination+')'
            return label+' [retained private evidence or external source; not bundled]'
        return re.sub(r'(!?)\[([^\]]*)\]\(([^)]+)\)', replace, text)
    for identity, path in docs:
        name = 'reports/'+identity+'.md'
        content = report_links(sanitize(path.read_text()), identity)
        content += '\n\nExport note: commands mentioning private packages describe the retained private workflow; this aggregate bundle includes only the verifier documented in its root README.\n'
        write(name,content.encode()); assets.append(dict(file=name,kind='report',text=content,source_sha256=sha(path.read_bytes())))
    for i,path in enumerate(figures):
        name = 'figures/'+str(i+1)+'-'+path.name
        content = sanitize(path.read_text())
        write(name,content.encode()); assets.append(dict(file=name,kind='figure',svg=content,source_sha256=sha(path.read_bytes())))
    for stem in SUPPLEMENTS:
        path=ROOT/'community/site/evidence'/(stem+'.zip')
        name='supplements/'+path.name
        raw=path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if archive.testzip() is not None:
                raise ValueError('Corrupt reviewed writing supplement: '+stem)
        write(name, raw)
        supplement_scopes = {
            'deepseek41-retained-mode2-20261109': 'Original reviewed same-L20 mode2 component ZIP, unchanged bytes; 20 source operands, 21 numeric/raw pairs and bounded numerical replay. Not the original crosslayer chain, native own scores or full-model/TP/training/acceptance.',
            'deepseek41-six-forward-selectors-20261109': 'Original reviewed six-forward rank0 tail4 sanitized aggregate ZIP, unchanged bytes; 30 selectors/120 dependent query cases/1740800 derived score coordinates. Four all-visible and two selective32768-key cohorts are distinct. Byte/CSV verification only; no independent numerical replay in ZIP, original-default unchunked32K closure, native own scores, full-history/allTP or acceptance.',
            'deepseek41-http128-20261109': 'Original reviewed HTTP128 package, unchanged bytes; includes bounded generated full-vocabulary arrays and a separate numerical verifier. Prompt-scalar/raw-response proof remains native-evidence bound.',
            'deepseek41-native-producer-key-cache-20261109': 'Original reviewed native producer-to-key/cache package, unchanged bytes; sampled projection inputs and full downstream arrays reconstruct actual FP16-scale cache bytes. Full private projection inputs are not included. FP32-scale reconstruction is a separate metric; no full-model or same-live-tensor equivalence.',
            'deepseek41-native-query-key-selection-20261109': 'Original reviewed retained-native query/key QLI package, unchanged bytes; all20 source arrays and74 actual numeric/raw pairs plus full derived reference score rows. ExactINT64 dot then actualFP16 scales/weights in FP64 is descriptive, not native own full scores. Tail4/all32-head serialized capsules, ratio2completed-M; L2mode3/L20mode1 only, no mode2consumer/fullmodel/training equivalence. 2Kall256candidate trivial branch is separate from32Ktop2048.',
        }
        assets.append(dict(file=name,kind='portable_supplement',source_sha256=sha(raw),
                           scope=supplement_scopes.get(stem, 'Original reviewed aggregate package, unchanged bytes; separate scope and verifier retained.')))
    result = dict(schema_version=1,snapshot_date='2026-11-09',scope='Paper-writing aggregates; explicit gaps retained',notes=NOTES,highlights=highlights(sources),sources=sources,numeric_fields=ledger,assets=assets)
    write('results.json',encoded(result))
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=['source_id','pointer','value','unit','kind']); writer.writeheader(); writer.writerows(ledger)
    write('values.csv',stream.getvalue().encode('utf-8-sig'))
    write('README.md',('# Paper-writing results / 论文写作结果\n\n'+'\n\n'.join(NOTES)+'\n\n读取 results.json 的 sources / numeric_fields；CSV 数值含精确来源 JSON Pointer。报告和图在 reports/ 与 figures/；新增 DeepSeek 报告、图和 CSV 位于 supplements/ 中逐字节保留的各独立 ZIP，解压后可运行各自 verifier。HTTP128 ZIP 包含有限范围的原生成词表数组及独立数值校验器；prompt 标量与原响应的一致性仍依赖原生独立证据。禁止忽略配置限定，禁止将缺项写成通过。\n\nRun `python -I -B verify.py` here to verify included bytes. Reports retain their original scientific context. The HTTP128 supplement ships bounded generated probability arrays and a numerical verifier; other supplements state their own aggregate or array scope. Model weights, full prompts/datasets, raw responses and private execution traces are excluded.\n').encode())
    write('verify.py',b'import hashlib,json,pathlib\nr=pathlib.Path(__file__).resolve().parent\nm=json.loads((r/"manifest.json").read_text())\nfor n,h in m.items():\n p=(r/n).resolve()\n assert p.is_relative_to(r),n\n assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n\nprint("PASS",len(m),"files; byte integrity only")\n')
    manifest={str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    write('manifest.json',encoded(manifest))
    with zipfile.ZipFile(output/'paper-writing-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p != output/'paper-writing-results.zip': z.write(p,str(p.relative_to(output)))
    print(json.dumps({'sources':len(sources),'numeric_fields':len(ledger),'reports':len(docs),'figures':len(figures),'zip_bytes':(output/'paper-writing-results.zip').stat().st_size}))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)


if __name__ == '__main__':
    raise SystemExit(main())
