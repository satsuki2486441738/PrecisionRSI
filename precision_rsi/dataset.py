"""build(files, index, output, held_out_families=(), splits=None) -> dataset receipt.

Input JSONL: {source: indexed task ID, messages: [{role, content}], outcome?: str}.
Outcome labels describe the source task, not correctness of every message.
Families stay in one deterministic train/validation/test split; external held-out
families are excluded. Exact conversations are deduplicated before writing.
"""
import json
from pathlib import Path

from .evidence import digest, verify
from .memory import read, write


def build(files, index, output, held_out_families=(), splits=None):
    if isinstance(index, (str, Path)):
        index = read(index)
    sources = {t['id']:t for t in index['tasks']}
    held_out, splits = set(held_out_families), splits or {}
    records, duplicates, verified = {}, 0, set()
    for filename in files:
        with Path(filename).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                source = sources.get(item['source'])
                if source is None:
                    raise ValueError('example must reference an indexed source task')
                if source['id'] not in verified:
                    verify(source)
                    verified.add(source['id'])
                family = source['family']
                if not family:
                    raise ValueError('training examples require an owner-declared task family')
                if family in held_out:
                    raise ValueError('example overlaps external held-out task family: ' + family)
                messages = item['messages']
                if (not isinstance(messages, list) or not messages or
                        not all(isinstance(m, dict) and m.get('role') in ('system', 'user', 'assistant', 'tool')
                        and isinstance(m.get('content'), str) and m['content'].strip() for m in messages)
                        or not any(m['role'] == 'assistant' for m in messages)):
                    raise ValueError('example requires role/content messages and an assistant response')
                if item.get('outcome', source['status']) != source['status']:
                    raise ValueError('example outcome disagrees with task evidence')
                bucket = int(digest(family)[:8], 16) % 10
                split = splits.get(family, 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train')
                if split not in ('train', 'validation', 'test'):
                    raise ValueError('unknown dataset split')
                key = digest(messages)
                value = dict(id=key, source=source['id'], family=family, split=split,
                             outcome=source['status'], label_scope='task', messages=messages)
                if key in records:
                    old = records[key]
                    if old['split'] != split or old['outcome'] != value['outcome']:
                        raise ValueError('duplicate conversation crosses splits or contradicts outcomes')
                    duplicates += 1
                else:
                    records[key] = value
    if not records:
        raise ValueError('dataset contains no examples')
    identity = digest(sorted(records.values(), key=lambda v:v['id']))
    output = Path(output) / identity
    output.mkdir(parents=True, exist_ok=True)
    counts, paths = {}, []
    for split in ('train', 'validation', 'test'):
        values = [records[key] for key in sorted(records) if records[key]['split'] == split]
        path = output / (split + '.jsonl')
        text = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
        if path.exists() and path.read_text() != text:
            raise ValueError('versioned dataset contents changed')
        if not path.exists():
            path.write_text(text)
        counts[split] = len(values)
        paths.append(str(path.resolve()))
    receipt = dict(id=identity, status='ready', counts=counts, duplicates=duplicates,
                   families=sorted({r['family'] for r in records.values()}), artifacts=paths)
    write(output / 'dataset.json', receipt)
    return receipt
