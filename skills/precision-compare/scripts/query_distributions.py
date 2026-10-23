#!/usr/bin/env python3
"""Compare distributions across hosts without moving full vocabulary arrays.
Call: inventory(contract, manifest); query(contract, manifest, peer_inventory); compare(contract, query_a, query_b)
Input: Source manifests, peer inventories and bound union-query responses.
Output: Distribution metrics from small exchanged queries; full arrays stay at source.
Requires: NumPy; each phase validates histories, source identities and top-100 union/tail coverage.

Details:
Uses the distribution contract/manifests in artifact_io.py. Three commands:
 inventory --contract C --manifest M --output NEW.json
 query --contract C --manifest M --peer-inventory I --output NEW.json
 compare --contract C --side-a QUERY_A --side-b QUERY_B --output NEW.json

Each host validates its source, publishes top-100 IDs plus SHA256 array identities,
then answers the union of its own and peer IDs from the original full logprobs.
Queries bind their score payload, exact complement log mass and reciprocal
source IDs in a checked query digest. Zero
probability is encoded as null in logprob arrays/tail. Compare uses the same
selected_metrics and summarize functions as compare_distributions.py. Arrays stay
on their source host; only inventories and queries need transfer. Sources must
remain immutable between inventory and query; changed identities are rejected.
API: inventory(contract_path, manifest_path), query(contract_path, manifest_path,
peer_inventory), compare(contract_path, query_a, query_b); final two APIs accept
already-loaded JSON objects. Outputs contain measured conditions and coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_io import canonical, contract, load_json, manifest, require, unique
from compare_distributions import coordinate, check_history, scores, lse, selected_metrics, summarize


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def verify_inventory(value, spec):
    require(isinstance(value,dict), 'source inventory must be an object')
    require(value.get('protocol')=='distribution-union-1', 'query protocol mismatch')
    require(value.get('contract_id')==digest(spec), 'query contract differs')
    core={k:value[k] for k in ('protocol','contract_id','side','positions','conditions')}
    require(value.get('source_id')==digest(core), 'source inventory identity differs')
    require(value['side'] in spec['expected_ranks'], 'undeclared query side')
    rows=unique(value['positions'],coordinate,'queried position')
    expected=unique(spec['positions'],coordinate,'expected position')
    require(set(rows)==set(expected) and bool(rows), 'query position coverage mismatch','not_covered')
    vocab=spec['vocab_size']
    for key,row in rows.items():
        require(row['history']==expected[key]['history'],'query history mismatch')
        ids=row['top100_ids']
        require(isinstance(ids,list) and len(ids)==100 and len(set(ids))==100 and
                all(type(t) is int and 0<=t<vocab for t in ids),'invalid top-100 IDs')
    return rows


def inventory(contract_path,manifest_path):
    spec=contract(contract_path,'distributions')
    require(type(spec.get('vocab_size')) is int and spec['vocab_size']>=100,'invalid vocabulary size')
    tolerance=spec.get('normalization_tolerance')
    require(type(tolerance) in (int,float) and math.isfinite(tolerance) and 0<tolerance<1,'invalid normalization tolerance')
    require(spec.get('renormalization') in ('none','full_vocabulary_fp64'),'invalid renormalization')
    data,base=manifest(manifest_path,spec)
    expected=unique(spec['positions'],coordinate,'expected position')
    observed=unique(data['positions'],coordinate,'position')
    require(set(expected)==set(observed) and bool(expected),'position coverage mismatch','not_covered')
    hashes={};positions=[]
    for key in sorted(expected):
        row=observed[key]
        check_history(row,spec['vocab_size']);check_history(expected[key],spec['vocab_size'])
        require(row['history']==expected[key]['history'],'history mismatch')
        values,_=scores(base,row,spec['vocab_size'],tolerance,spec['renormalization'])
        top=np.lexsort((np.arange(len(values)),-values))[:100]
        path=(base/row['scores']['path']).resolve()
        if path not in hashes:hashes[path]=file_digest(path)
        positions.append(dict(sample=key[0],position=key[1],history=row['history'],top100_ids=top.tolist(),
            source=dict(sha256=hashes[path],key=row['scores'].get('key'),row=row['scores'].get('row',0))))
    result=dict(protocol='distribution-union-1',contract_id=digest(spec),side=data['side'],positions=positions,
                conditions=data['_condition_check'])
    result['source_id']=digest(result)
    return result


def query(contract_path,manifest_path,peer_inventory):
    spec=contract(contract_path,'distributions')
    own=inventory(contract_path,manifest_path)
    peer=verify_inventory(peer_inventory,spec)
    require(own['side']!=peer_inventory['side'],'queries require distinct sides')
    data,base=manifest(manifest_path,spec)
    observed=unique(data['positions'],coordinate,'position')
    rows=[]
    for entry in own['positions']:
        key=coordinate(entry)
        union=np.union1d(entry['top100_ids'],peer[key]['top100_ids'])
        values,mass=scores(base,observed[key],spec['vocab_size'],spec['normalization_tolerance'],spec['renormalization'])
        mask=np.ones(len(values),dtype=bool);mask[union]=False
        tail=lse(values[mask])
        rows.append(dict(sample=key[0],position=key[1],union=union.tolist(),
            logprobs=[None if v==-math.inf else float(v) for v in values[union]],
            tail_logmass=None if tail==-math.inf else tail,normalization_mass=mass))
    # Check again after row reads so a changed source cannot keep its earlier identity.
    require(inventory(contract_path,manifest_path)['source_id']==own['source_id'],'source changed during query')
    result=dict(inventory=own,peer_source_id=peer_inventory['source_id'],positions=rows)
    result['query_id']=digest(result)
    return result


def compare(contract_path,query_a,query_b):
    spec=contract(contract_path,'distributions')
    for q in (query_a,query_b):
        require(isinstance(q,dict) and q.get('query_id')==digest({k:q[k] for k in ('inventory','peer_source_id','positions')}),
                'query payload identity differs')
    inventories=[q['inventory'] for q in (query_a,query_b)]
    indexed=[verify_inventory(i,spec) for i in inventories]
    require(inventories[0]['side']!=inventories[1]['side'],'query side IDs must differ')
    require(query_a['peer_source_id']==inventories[1]['source_id'] and
            query_b['peer_source_id']==inventories[0]['source_id'],'peer source binding differs')
    records=[unique(q['positions'],coordinate,'query row') for q in (query_a,query_b)]
    require(set(records[0])==set(records[1])==set(indexed[0]),'query coverage mismatch','not_covered')
    rows=[]
    for key in sorted(records[0]):
        a,b=[r[key] for r in records]
        union=sorted(set(indexed[0][key]['top100_ids'])|set(indexed[1][key]['top100_ids']))
        require(a['union']==b['union']==union,'query union differs')
        def values(row):
            result=row['logprobs']+[row['tail_logmass']]
            require(all(v is None or (type(v) in (int,float) and math.isfinite(v) and v<=0) for v in result),'invalid query logprob')
            mass=row['normalization_mass']
            require(type(mass) in (int,float) and math.isfinite(mass) and abs(mass-1)<=spec['normalization_tolerance'],'invalid original probability mass')
            return [-math.inf if v is None else v for v in result]
        metrics=selected_metrics(union,values(a),values(b),indexed[0][key]['top100_ids'],indexed[1][key]['top100_ids'],
            normalization_tolerance=spec['normalization_tolerance'])
        rows.append(dict(sample=key[0],position=key[1],**metrics))
    return dict(kind='distributions',schema_version=2,status='no_thresholds',thresholds=None,
        sides=dict(side_a=inventories[0]['side'],side_b=inventories[1]['side']),
        position_count=len(rows),positions=rows,aggregate=summarize(rows),
        source_ids=[i['source_id'] for i in inventories],conditions=[i['conditions'] for i in inventories],
        normalization_tolerance=spec['normalization_tolerance'],renormalization=spec['renormalization'])


def main():
    parser=argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=('inventory','query','compare'))
    parser.add_argument('--contract',required=True)
    for name in ('manifest','peer-inventory','side-a','side-b'):parser.add_argument('--'+name)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    required={'inventory':['manifest'],'query':['manifest','peer_inventory'],'compare':['side_a','side_b']}[args.command]
    for field in required:
        if getattr(args,field) is None:parser.error('--'+field.replace('_','-')+' is required')
    try:
        if args.command=='inventory':result=inventory(args.contract,args.manifest)
        elif args.command=='query':result=query(args.contract,args.manifest,load_json(args.peer_inventory))
        else:result=compare(args.contract,load_json(args.side_a),load_json(args.side_b))
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
        print(str(exc),file=sys.stderr);return 1
    try:
        with Path(args.output).open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
