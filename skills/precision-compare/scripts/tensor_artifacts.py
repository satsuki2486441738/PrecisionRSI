#!/usr/bin/env python3
"""Export logical tensor shards and compare explicit replica views.

export_manifest(contract_path,capture_path,view=None) returns a portable manifest.
Capture JSON: side, common, expected_ranks, stage, optional events and
ranks:[{rank,environment:JSON_path,status:JSON_path,inventory:JSON_path}].
Each inventory is a list of {name,shape,present,shards:[{offset,value|before+after}]}.
Observation-stage entries also carry contract-matching coordinates:{...}.
Array descriptors follow artifact_io.py and resolve relative to the inventory.
Adapters supply logical names/shapes/offsets; no framework topology is inferred.
A shard may carry view:str. Without a selected view, duplicate intervals are
collapsed only after equal stored values are verified. Differing replicas require
explicit views. A selected view retains untagged shards plus matching view shards.
There are no fixed rank-count or storage-dtype assumptions.

compare_views(contract_path,pairs,base=Path('.')) compares pairs=[{view:str,
side_a:manifest_path,side_b:manifest_path, optional side_a_view/side_b_view}].
Exported views are bound to the pair IDs (default: view); an unselected export
uses view=logical. Each manifest supplies one complete
logical view, retaining all rank execution records. Results remain per-view;
no replica averaging or combined canonical tensor is implied.

CLI: export --contract C --capture CAPTURE --view optional --output NEW.json
     compare-views --contract C --pairs PAIRS.json --output NEW.json
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_io import Array, blocked_status, canonical, contract, integer, load_json, require, shape_size
from conditions import execution_settings
from compare_tensors import compare, inventory


def equal_shards(left,right):
    for name in ('value','before','after'):
        require((name in left)==(name in right),'replica encodings differ')
        if name not in left:continue
        a,b=[Array(Path('.'),v[name]) for v in (left,right)]
        if a.size!=b.size:return False
        for start in range(0,a.size,1_048_576):
            n=min(1_048_576,a.size-start)
            if not np.array_equal(a.read(start,n),b.read(start,n)):return False
    return True


def export_manifest(contract_path,capture_path,view=None):
    require(view is None or (isinstance(view,str) and bool(view)), 'view must be a nonempty ID')
    spec=contract(contract_path,'tensors');expected=inventory(spec)
    capture=load_json(capture_path);base=Path(capture_path).resolve().parent
    data={key:capture[key] for key in ('side','common','expected_ranks','stage')}
    require(data['stage']==spec['stage'],'capture stage differs')
    data.update(schema_version=spec['schema_version'],kind='tensors',ranks=[])
    if 'events' in capture:data['events']=capture['events']
    entries={name:[] for name in expected}
    for rank in capture['ranks']:
        data['ranks'].append(dict(rank=rank['rank'],**{key:str((base/rank[key]).resolve()) for key in ('environment','status')}))
        path=(base/rank['inventory']).resolve()
        rows=load_json(path)
        require(isinstance(rows,list),'tensor inventory must be a list')
        seen=set()
        for row in rows:
            name=row['name'];require(name in expected and name not in seen,'unexpected/duplicate logical parameter')
            seen.add(name)
            require(row['shape']==expected[name]['shape'] and type(row.get('present')) is bool and row['present']==expected[name]['present'],'logical tensor inventory differs')
            if data['stage']=='observations':
                require(canonical(row.get('coordinates'))==canonical(expected[name]['coordinates']),
                        'observation coordinates differ')
            require(isinstance(row.get('shards'),list),'logical shards required')
            for shard in row['shards']:
                if view is not None and shard.get('view',view)!=view:continue
                require(integer(shard.get('offset')),'invalid logical offset')
                present=[key for key in ('value','before','after') if key in shard]
                require(present in (['value'],['before','after']),'invalid shard encoding')
                item=dict(offset=shard['offset'])
                for key in present:
                    descriptor=dict(shard[key]);descriptor['path']=str((path.parent/descriptor['path']).resolve())
                    item[key]=descriptor
                entries[name].append(item)
    execution_settings(spec,data,base,load_json,require_actual=spec['schema_version']==2 or bool(spec.get('side_conditions')))
    tensors=[]
    for name,shards in entries.items():
        item=expected[name];selected=[];end=0
        for shard in sorted(shards,key=lambda s:s['offset']):
            array=Array(Path('.'),shard.get('value',shard.get('before')))
            if selected and shard['offset']==selected[-1]['offset']:
                require(equal_shards(selected[-1],shard),'differing replicas require explicit view selection')
                continue
            require(shard['offset']==end,'logical shard gap/overlap','not_covered')
            selected.append(shard);end+=array.size
        require((not item['present'] and not selected) or (item['present'] and end==shape_size(item['shape'])),
                'incomplete selected tensor view','not_covered')
        tensors.append({**item,'shards':selected})
    data['view']=view if view is not None else 'logical'
    data['tensors']=tensors
    return data


def compare_views(contract_path,pairs,base=Path('.')):
    require(isinstance(pairs,list) and bool(pairs),'view pairs required')
    results={};seen=set()
    for pair in pairs:
        name=pair['view']
        require(isinstance(name,str) and name and name not in results,'unique view IDs required')
        paths=[(base/pair[key]).resolve() for key in ('side_a','side_b')]
        require(frozenset(paths) not in seen, 'duplicate capture pair across views')
        seen.add(frozenset(paths))
        for side,path in zip(('side_a','side_b'),paths):
            observed=load_json(path)
            require(isinstance(observed,dict), 'view manifest must be an object')
            require(observed.get('view')==pair.get(side+'_view',name), 'exported replica view differs from selected view')
        results[name]=compare(contract_path,*paths)
    status='no_thresholds' if all(r['status']=='no_thresholds' for r in results.values()) else blocked_status(results.values())
    return dict(kind='tensor-views',status=status,thresholds=None,view_count=len(results),views=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('export','compare-views'))
    parser.add_argument('--contract',required=True)
    parser.add_argument('--capture');parser.add_argument('--view');parser.add_argument('--pairs')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='export' and args.capture is None:parser.error('--capture is required')
    if args.command=='compare-views' and args.pairs is None:parser.error('--pairs is required')
    try:
        if args.command=='export':result=export_manifest(args.contract,args.capture,args.view)
        else:
            path=Path(args.pairs).resolve();result=compare_views(args.contract,load_json(path),path.parent)
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as exc:
        print(str(exc),file=sys.stderr);return 1
    try:
        with Path(args.output).open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 2
    return 0 if args.command=='export' or result['status']=='no_thresholds' else 1


if __name__=='__main__':raise SystemExit(main())
