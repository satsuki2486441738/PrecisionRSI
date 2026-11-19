#!/usr/bin/env python3
"""Generate paired inference, SFT and RL inputs for simple synthetic events.

Usage: python -B generate_precision_batch.py --output NEW.json
Uses the standard library only. Identical arguments produce identical bytes.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path


EVENTS = (
    ('coin', 'A covered coin has landed.', 'heads', 'tails'),
    ('draw', 'One chip was drawn from an opaque bag.', 'red', 'blue'),
    ('spinner', 'A two-section spinner has stopped behind a screen.', 'star', 'circle'),
    ('die', 'A die was rolled under a cup.', 'odd', 'even'),
)
CHOICES = ('A', 'B')


def score_choice(completion, target):
    """Return the binary reward and whether the completion is a valid choice."""
    choice = completion.strip()
    return int(choice == target), choice in CHOICES


def build_batch(count=100, seed=17, group_size=4):
    if count < 1 or group_size < 2:
        raise ValueError('count must be positive and group size must be at least two')

    batch = dict(schema_version=1, seed=seed, count=count, group_size=group_size,
                 inference=[], sft=[], rl_online=[], rl_replay=[], answer_key=[])
    for index in range(count):
        kind, setup, *outcomes = EVENTS[index % len(EVENTS)]
        key = f'numericbridge-events-v1:{seed}:{index}'.encode('ascii')
        event_id = f'NB-{index + 1:04d}-{sha256(b"id:" + key).hexdigest()[:8]}'
        result_index = sha256(b'result:' + key).digest()[0] & 1
        target, other = CHOICES[result_index], CHOICES[1 - result_index]
        result = outcomes[result_index]
        header = f'Event {event_id}. {setup}'
        if (index // len(EVENTS)) % 2:
            header += ' This is a fictional practice round recorded by the event judge.'
        choices = f'A means {outcomes[0]}; B means {outcomes[1]}. Reply with A or B only.'
        visible = f'{header} The recorded outcome is {result}. {choices}'
        hidden = f'{header} The outcome is hidden. Guess the outcome. {choices}'

        batch['inference'].append(dict(id=event_id, prompt=visible, continuations=list(CHOICES)))
        batch['sft'].append(dict(id=event_id, prompt=visible, completion=target))
        batch['rl_online'].append(dict(id=event_id, prompt=hidden))
        batch['answer_key'].append(dict(id=event_id, event=kind, outcome=result, target=target))
        for mode in ('mixed', 'all_correct', 'all_wrong'):
            if mode == 'mixed':
                completions = [target if i % 2 == 0 else other for i in range(group_size)]
            else:
                completions = [target if mode == 'all_correct' else other] * group_size
            rewards = [score_choice(text, target)[0] for text in completions]
            batch['rl_replay'].append(dict(id=event_id, group=mode,
                                           completions=completions, rewards=rewards))
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('--output', type=Path, required=True, help='new JSON file')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--group-size', type=int, default=4)
    args = parser.parse_args()
    try:
        batch = build_batch(args.count, args.seed, args.group_size)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(batch, stream, indent=2)
            stream.write('\n')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
