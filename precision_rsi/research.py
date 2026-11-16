"""Durable researcher state; no full trace or evaluator inputs in the public view.

CLI: python -m precision_rsi.research VIEW state
     python -m precision_rsi.research VIEW history [--start N --count 5]
         [--newest] [--outcome accepted|unaccepted|incomplete] [--candidate ID]
     python -m precision_rsi.research VIEW notes|plan [--set JSON_FILE]
     python -m precision_rsi.research ROOT reconcile TURN --reason TEXT
Notes: {observations:[], hypotheses:[{claim,evidence_rounds:[]}], uncertainties:[]}.
Plan: {next_experiments:[{intervention,falsified_if,evidence_rounds:[]}]}.
Each document is optional, bounded to 4096 UTF-8 bytes. Round references must exist.
Claims remain researcher judgments; ledger outcomes are controller observations.
"""
import argparse
import json
from pathlib import Path

from .evidence import digest
from .memory import read, workspace_lock, write


def _measurements(value):
    """Keep numeric observations and unknowns, never host strings or arbitrary keys."""
    fields = {'correct','seconds','spent_seconds','cost','quality','budget','usage',
              'input_tokens','output_tokens','total_tokens','cached_input_tokens','steps',
              'baseline','candidate','construction','amortize_over_tasks',
              'execution','including_construction'}
    if not isinstance(value, dict):
        return None
    return {k:(_measurements(v) if isinstance(v, dict) else v) for k,v in value.items()
            if k in fields and (isinstance(v, dict) or v is None or type(v) in (int,float,bool))}


def _file(root, name):
    path = Path(root)/name
    if Path(root).is_symlink() or path.is_symlink():
        raise ValueError('research state must not be a symlink')
    return path


def history(view, start=0, count=5, *, newest=False, outcome=None, candidate=None):
    """Page matching rounds; offsets refer to the filtered, ordered result."""
    if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 5:
        raise ValueError('history requires start>=0 and count=1..5')
    if type(newest) is not bool or outcome not in (None, 'accepted', 'unaccepted', 'incomplete'):
        raise ValueError('invalid history order or outcome')
    if candidate is not None and (not isinstance(candidate, str) or len(candidate) != 64 or
                                  any(c not in '0123456789abcdef' for c in candidate)):
        raise ValueError('candidate must be a lowercase SHA-256 ID')
    ledger = read(_file(view, 'ledger.json'))
    matches = [row for row in ledger if
        (outcome is None or
         outcome == 'accepted' and row['accepted'] is True or
         outcome == 'unaccepted' and row['accepted'] is False or
         outcome == 'incomplete' and row['status'] != 'completed') and
        (candidate is None or candidate in row['candidates'])]
    if newest:
        matches.reverse()
    end = min(start+count, len(matches))
    return dict(total=len(matches), ledger_total=len(ledger), records=matches[start:end],
                next_start=end if end < len(matches) else None)


def state(view):
    """One bounded handoff view; full history and native traces stay off-prefix."""
    ledger_path = _file(view, 'ledger.json')
    ledger = read(ledger_path) if ledger_path.exists() else []
    accepted = [row['round'] for row in ledger if row['accepted'] is True]
    documents = {}
    for kind in ('notes', 'plan'):
        if _file(view, kind+'.json').exists():
            try:
                documents[kind] = document(view, kind)['judgments']
            except (ValueError, TypeError) as exc:
                documents[kind] = dict(error=str(exc)[:200])
    result = dict(summary=dict(rounds=len(ledger), accepted_rounds=len(accepted),
        incomplete_rounds=sum(row['status'] != 'completed' for row in ledger),
        last_accepted_round=accepted[-1] if accepted else None), documents=documents)
    checkpoint = _file(view, 'handoff.json')
    if checkpoint.exists():
        result['handoff'] = read(checkpoint)
    return result


def document(view, kind, value=None, *, save=True):
    """Validate judgments against observed rounds; save=False performs no write."""
    if kind not in ('notes', 'plan'):
        raise ValueError('expected notes or plan')
    path = _file(view, kind+'.json')
    with workspace_lock(Path(view)):
        if value is None:
            value = read(path) if path.exists() else {}
        elif not isinstance(value, dict):
            raise ValueError('document must be an object')
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > 4096:
            raise ValueError('document exceeds 4096 bytes')
        allowed = {'next_experiments'} if kind == 'plan' else {'observations','hypotheses','uncertainties'}
        if set(value)-allowed or any(not isinstance(v, list) for v in value.values()):
            raise ValueError('unknown document fields or non-list values')
        ledger = read(_file(view, 'ledger.json'))
        references = set()
        for field, items in value.items():
            for item in items:
                if field in ('observations', 'uncertainties'):
                    if not isinstance(item, str):
                        raise ValueError('observations and uncertainties must be strings')
                    continue
                required = {'claim','evidence_rounds'} if field == 'hypotheses' else {
                    'intervention','falsified_if','evidence_rounds'}
                if not isinstance(item, dict) or set(item) != required:
                    raise ValueError('invalid hypothesis or experiment fields')
                if any(not isinstance(item[k], str) or not item[k].strip() for k in required-{'evidence_rounds'}):
                    raise ValueError('claim and experiment text required')
                refs = item['evidence_rounds']
                if not isinstance(refs, list) or len(refs) > 16 or any(
                        type(n) is not int or not 1 <= n <= len(ledger) for n in refs):
                    raise ValueError('unknown evidence round')
                references.update(refs)
        if len(references) > 16:
            raise ValueError('document references more than 16 rounds; use paged history')
        # Write only after the complete document has been validated.
        if save and (not path.exists() or read(path) != value):
            write(path, value)
        return dict(judgments=value, evidence=[ledger[n-1] for n in sorted(references)])


def save_handoff(view, value):
    """Accept changed judgments from the normal task result, with no extra call."""
    if not isinstance(value, dict) or set(value)-{'notes', 'plan'}:
        raise ValueError('handoff supports notes and plan')
    # Validate the entire submission before changing either document. Reapplying
    # after an interrupted pair of atomic writes is idempotent.
    for kind, content in value.items():
        if not isinstance(content, dict):
            raise ValueError('handoff documents must be objects')
        document(view, kind, content, save=False)
    for kind, content in value.items():
        document(view, kind, content)


class Research:
    """Controller owns private receipts/cursor and publishes a redacted full ledger."""
    def __init__(self, root, binding=None):
        self.root = Path(root)
        self.view, self.private = self.root/'view', self.root/'private'
        if self.root.is_symlink():
            raise ValueError('research root must not be a symlink')
        for path in (self.view, self.private):
            if path.is_symlink():
                raise ValueError('research directory must not be a symlink')
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        bound = _file(self.private, 'binding.json')
        if binding is None and not bound.exists():
            raise ValueError('existing research binding required')
        identity = dict(key=digest(binding)) if binding is not None else read(bound)
        if bound.exists() and read(bound) != identity:
            raise ValueError('research binding changed; use a new run')
        if not bound.exists():
            write(bound, identity)

    def sync(self, rounds):
        # Never copy candidate source, native traces, host paths or evaluator configs.
        ledger = []
        for number, entry in enumerate(rounds, 1):
            feedback = entry['feedback']
            result = entry['result']
            ledger.append(dict(round=number, status=result.get('status'),
                accepted=entry['accepted'], repeated=entry['repeated'],
                no_update=result.get('no_update', False), candidates=result.get('candidates', []),
                construction=_measurements(feedback.get('construction')),
                evaluation=_measurements(feedback.get('evaluation')),
                measurements=_measurements(feedback.get('measurements', {}))))
        write(_file(self.view, 'ledger.json'), ledger)

    def context(self, python):
        """Reattach bounded current judgments, including after native compaction."""
        current = state(self.view)
        return dict(view=str(self.view), interface=[python, '-m', 'precision_rsi.research', str(self.view)],
            operations='state; history [--start N --count 5] [--newest] '
                       '[--outcome accepted|unaccepted|incomplete] [--candidate ID]; '
                       'notes|plan [--set JSON_FILE]',
            schemas=dict(
                notes=dict(observations=['text'], uncertainties=['text'],
                    hypotheses=[dict(claim='text', evidence_rounds=[])]),
                plan=dict(next_experiments=[dict(intervention='text', falsified_if='text', evidence_rounds=[])])),
            semantics='Ledger is authoritative; notes/plan are optional judgments, 4096 bytes each. '
                      '--set replaces the document. evidence_rounds contains existing round numbers '
                      '(at most 16); [] means no supporting observed round yet. '
                      'History stays queryable after compact. Attached documents are current; '
                      'query only missing history. Before handoff, replace notes/plan only when '
                      'conclusions or next actions changed.',
            summary=current['summary'], documents=current['documents'],
            handoff=current.get('handoff'))

    def checkpoint(self, turn, run, *, phase='construction', status='running',
                   usage=None, judgment_update='not_submitted'):
        """Publish host-observed progress, including when no judgments were saved.

        Only top-level artifact filenames are indexed; existence is not validation.
        Native sessions, commands, evaluator paths and artifact bodies stay private.
        """
        if phase not in ('construction', 'evaluation', 'review') or status not in (
                'running', 'completed', 'failed', 'interrupted', 'needs_action'):
            raise ValueError('invalid checkpoint phase or status')
        run = Path(run)
        folder = run/'artifacts'
        names = sorted(p.name for p in folder.iterdir() if p.is_file() and not p.is_symlink()) \
            if folder.is_dir() else []
        proposal = (run/'proposals.json').is_file()
        action = ('inspect_retained_work' if status in ('failed', 'interrupted', 'needs_action') else
                  'await_evaluation' if phase == 'evaluation' else
                  'review_outcome' if phase == 'review' else
                  'verify_saved_proposal' if proposal else 'complete_proposal')
        value = dict(origin='controller', turn=turn, phase=phase, status=status,
            reported_usage=_measurements(usage), proposal_present=proposal,
            artifact_scope='unverified_top_level_files', artifact_count=len(names),
            artifacts=[name[:120] for name in names[:12]], next_action=action,
            judgment_update=judgment_update)
        path = _file(self.view, 'handoff.json')
        if not path.exists() or read(path) != value:
            write(path, value)
        return value

    def begin(self, turn, binding):
        if not turn or not turn.replace('-', '').isalnum():
            raise ValueError('invalid researcher turn')
        identity = digest(binding)
        receipt = _file(self.private, 'turn-'+turn+'.json')
        active = _file(self.private, 'active.json')
        if receipt.exists():
            value = read(receipt)
            if value['key'] != identity:
                raise ValueError('turn inputs changed')
            if active.exists() and read(active)['turn'] == turn:
                active.unlink()
            return value
        if active.exists():
            raise ValueError('unfinished researcher turn; reconcile explicitly before continuing')
        write(active, dict(turn=turn, key=identity))
        return None

    def complete(self, turn, **result):
        active = _file(self.private, 'active.json')
        value = read(active)
        if value['turn'] != turn:
            raise ValueError('researcher turn changed')
        value.update(result)
        write(_file(self.private, 'turn-'+turn+'.json'), value)
        active.unlink()
        return value

    def reconcile(self, turn, reason):
        """Operator explicitly abandons an unfinished turn; never retries that turn."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('reconciliation reason required')
        return self.complete(turn, status='interrupted', reason=reason)

    def files(self):
        """Visible seed separately from operator-only native session/receipts."""
        visible, private = {}, {}
        for name in ('ledger.json','notes.json','plan.json','handoff.json'):
            path = _file(self.view, name)
            if path.is_file():
                visible['research/'+name] = path
        for path in self.private.rglob('*'):
            if path.is_symlink():
                raise ValueError('symlink in private research state')
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                private['research/'+path.relative_to(self.private).as_posix()] = path
        return visible, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('location', help='VIEW for memory tools; ROOT for operator reconciliation')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('state', help='bounded summary and current notes/plan in one read')
    page = sub.add_parser('history')
    page.add_argument('--start', type=int, default=0)
    page.add_argument('--count', type=int, default=5)
    page.add_argument('--newest', action='store_true')
    page.add_argument('--outcome', choices=('accepted', 'unaccepted', 'incomplete'))
    page.add_argument('--candidate')
    for name in ('notes', 'plan'):
        sub.add_parser(name).add_argument('--set', dest='source')
    recovery = sub.add_parser('reconcile', help='operator-only: confirm unfinished process has stopped')
    recovery.add_argument('turn')
    recovery.add_argument('--reason', required=True)
    args = parser.parse_args()
    if args.command == 'reconcile':
        research = Research(args.location)
        with workspace_lock(research.private):
            result = research.reconcile(args.turn, args.reason)
    else:
        result = (state(args.location) if args.command == 'state' else
                  history(args.location, args.start, args.count, newest=args.newest,
                          outcome=args.outcome, candidate=args.candidate) if args.command == 'history' else
                  document(args.location, args.command, read(args.source) if args.source else None))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
