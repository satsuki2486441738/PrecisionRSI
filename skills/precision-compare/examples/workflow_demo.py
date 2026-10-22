#!/usr/bin/env python3
"""Run a native Torch adaptation -> compare -> repair -> revalidation example.

Requires Torch and NumPy. Creates a fresh compare-workflowdemo-UTC case and a
shared workflow plan. A controlled mean-versus-sum defect in a two-step scalar
SFT objective routes comparison to repair. This demonstration applies the known
one-line correction; it does not invoke or benchmark an agent. The second run
reuses environment preparation and verifies the corrected original trajectory;
a third run must execute zero native commands. --prepare-only creates the task
for an agent to complete instead. Inputs, baseline results and logs are retained.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(TOOLS))
from workflow import run_workflow
from resource_catalog import register, find

DRIVER = '''import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, TOOL_PATH)
from runtime_probe import inspect_runtime, runtime_identity
from compare_tensors import Stats
from objective import objective

torch.set_num_threads(1)
mode=sys.argv[1]
out=Path(sys.argv[2])
data=json.loads(Path('data.json').read_text())
def native():
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    optimizer=torch.optim.SGD(model.parameters(),lr=data['lr'])
    x=torch.ones(len(data['targets']),1)
    y=torch.tensor(data['targets']).reshape(-1,1)
    gradients=[];parameters=[];updates=[]
    for step in range(data['steps']):
        optimizer.zero_grad(set_to_none=True)
        before=model.weight.detach().clone()
        objective(model(x),y).backward()
        gradients.append(float(model.weight.grad.item()))
        optimizer.step()
        parameters.append(float(model.weight.item()))
        updates.append(float((model.weight.detach()-before).item()))
    return {k:np.asarray(v) for k,v in [('gradients',gradients),('parameters',parameters),('updates',updates)]}
def expected():
    # Analytic derivative of mean((w-target)**2)/2, independent of objective.py.
    w=0.;g=[];p=[];u=[]
    for step in range(data['steps']):
        gradient=w-float(np.mean(data['targets']))
        delta=-data['lr']*gradient
        w+=delta;g.append(gradient);p.append(w);u.append(delta)
    return dict(gradients=np.asarray(g),parameters=np.asarray(p),updates=np.asarray(u))
def compare(values):
    metrics={}
    for name,oracle in expected().items():
        stats=Stats();stats.add(oracle,values[name]);metrics[name]=stats.report()
    maximum=max(v['max_abs']['value'] for v in metrics.values())
    return dict(metrics=metrics,correct=maximum<=data['regression_tolerance'],
                max_abs=maximum,coordinates=data['steps']*len(metrics),
                tolerance_basis='FP32 scalar SGD against analytic objective')
if mode=='adapt':
    smoke=torch.nn.Linear(1,1)(torch.ones(1,1))
    report=inspect_runtime(dict(imports=['torch','torch._C','numpy'],symbols=['torch.nn:Linear']))
    report['device']=str(smoke.device);report['shape']=list(smoke.shape)
    out.with_name('runtime-full.json').write_text(json.dumps(report))
    identity=runtime_identity(report);identity.update(status=report['status'],device=report['device'])
    out.write_text(json.dumps(identity))
elif mode=='collect':
    np.savez(out,**native())
elif mode=='compare':
    with np.load(sys.argv[3]) as values:report=compare(values)
    out.write_text(json.dumps(report))
elif mode=='regression':
    report=compare(native());out.write_text(json.dumps(report))
    if not report['correct']:raise SystemExit(1)
'''


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(output_root):
    root = Path(output_root).resolve() / datetime.now(timezone.utc).strftime('compare-workflowdemo-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True)
    (root / 'driver.py').write_text('TOOL_PATH=' + repr(str(TOOLS)) + '\n' + DRIVER)
    (root / 'objective.py').write_text('def objective(prediction, targets):\n    return ((prediction-targets)**2).sum()/2\n')
    write(root / 'data.json', dict(targets=[1., 3.], lr=0.1, steps=2, regression_tolerance=1e-6))
    write(root / 'environment.json', dict(python=sys.executable, framework='torch', device='cpu'))
    base = dict(argv=[], outputs={}, timeout=30, inputs=['driver.py'])
    stages = [
        dict(base, id='adapt', skill='framework-adapt',
             inputs=['driver.py', 'environment.json', str(TOOLS / 'runtime_probe.py')],
             argv=[sys.executable, '-B', 'driver.py', 'adapt', '{run}/runtime.json'],
             outputs={'runtime': 'runtime.json'}, checks=[dict(output='runtime', pointer='/status', equals='completed')]),
        dict(base, id='collect', skill='precision-compare', depends_on=['adapt'],
             inputs=['driver.py', 'objective.py', 'data.json'],
             argv=[sys.executable, '-B', 'driver.py', 'collect', '{run}/trajectory.npz'],
             outputs={'trajectory': 'trajectory.npz'}),
        dict(base, id='compare', skill='precision-compare', depends_on=['collect'],
             inputs=['driver.py', 'data.json', str(TOOLS / 'compare_tensors.py')],
             argv=[sys.executable, '-B', 'driver.py', 'compare', '{run}/metrics.json', '{artifact:collect:trajectory}'],
             outputs={'metrics': 'metrics.json'}, checks=[dict(output='metrics', pointer='/correct', equals=True)],
             on_check_failure='precision-repair')]
    plan = dict(schema_version=1, id='native-training', budget_seconds=120,
                workload=dict(model='torch.nn.Linear(1,1)', task='two-step full-parameter SFT', sides=['cpu']),
                stages=stages)
    write(root / 'task.json', plan)
    write(root / 'request.json', dict(task='Run the supplied two-step Torch SFT workload, compare with its analytic objective, fix confirmed correctness problems and verify the original trajectory.'))
    return root, plan


def demonstrate(output_root, prepare_only=False):
    root, plan = prepare(output_root)
    if prepare_only:
        return dict(case=str(root), plan=str(root / 'task.json'))
    before = run_workflow(root / 'task.json', root / 'session')
    write(root / 'before.json', before)
    assert before['next_action']['skill'] == 'precision-repair', before
    prepared = before['stages'][0]
    runtime=json.loads(Path(prepared['outputs']['runtime']['path']).read_text())
    resources=dict(declaration=str(root/'environment.json'), runtime=prepared['outputs']['runtime']['path'],
                   python=sys.executable)
    for index, module in enumerate(runtime['modules'].values()):
        if module['path']:resources['module'+str(index)]=module['path']
    entry = dict(id='native-torch-cpu', kind='environment',
                 properties=dict(framework='torch', device='cpu', torch=runtime['modules']['torch']['version']),
                 paths=resources, evidence=str(Path(prepared['run'])/'command.json'))
    write(root/'environment-entry.json', entry)
    catalog=root/'resources/catalog.json'
    register(catalog, root/'environment-entry.json')
    assert len(find(catalog, 'environment', dict(framework='torch', device='cpu')))==1
    baseline = Path(before['next_action']['outputs']['metrics']['path'])
    baseline_bytes = baseline.read_bytes()
    (root / 'objective-original.py').write_bytes((root / 'objective.py').read_bytes())
    (root / 'objective.py').write_text((root / 'objective.py').read_text().replace('.sum()', '.mean()'))
    regression = dict(id='regression', skill='precision-repair', depends_on=['adapt'],
        inputs=['driver.py', 'objective.py', 'data.json', str(TOOLS / 'compare_tensors.py')], timeout=30,
        argv=[sys.executable, '-B', 'driver.py', 'regression', '{run}/regression.json'],
        outputs={'regression': 'regression.json'}, checks=[dict(output='regression', pointer='/correct', equals=True)])
    plan['stages'].insert(2, regression)
    plan['stages'][-1]['depends_on'].append('regression')
    write(root / 'task.json', plan)
    after = run_workflow(root / 'task.json', root / 'session')
    write(root / 'after.json', after)
    assert after['status'] == 'completed' and after['stages'][0]['reused'], after
    replay = run_workflow(root / 'task.json', root / 'session')
    assert all(s['reused'] for s in replay['stages']) and replay['attempts'] == after['attempts'], replay
    assert baseline.read_bytes() == baseline_bytes
    result = dict(case=str(root), routed_to_repair=True, before_max_abs=json.loads(baseline_bytes)['max_abs'],
                  after_max_abs=json.loads(Path(after['stages'][-1]['outputs']['metrics']['path']).read_text())['max_abs'],
                  preparation_reused=True, resource_discovered=True, replay_native_commands=0, attempts=after['attempts'])
    write(root / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(demonstrate(args.output_root, args.prepare_only)))


if __name__ == '__main__':
    main()
