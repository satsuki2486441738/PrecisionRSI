"""Exercise the shipped DSH policy through real headless sessions, with a mock LLM.

Set DSH_BIN to the installed @deepseek-ai/dsh launcher to enable integration.
The temporary DSH_HOME never adopts production sessions or credentials.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PATCH = Path(__file__).resolve().parents[1] / 'research/dsh-compaction.patch.json'


class DshCompactionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DSH_BIN'), 'optional DSH integration')
    def test_automatic_checkpoint_preserves_state_and_continues(self):
        requests = []
        diagnostics = []
        marker = 'checkpoint_marker=RSI_7391'

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                text = json.dumps(body['messages'])
                summary = 'You are now acting as a compaction engine' in text
                answer = (f'## Current Work\n- {marker}\n## Next Step\n- Continue.'
                          if summary else f'ACK {marker}')
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for delta, finish in [({'role': 'assistant', 'content': answer}, None),
                                      ({}, 'stop')]:
                    event = {'id': 'mock', 'object': 'chat.completion.chunk',
                             'created': 0, 'model': 'compact-test',
                             'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix='rsi-compact-') as tmp:
                root = Path(tmp)
                model_patch = root / 'model.json'
                model_patch.write_text(json.dumps([
                    {'id': 'llm-deepseek', 'disabled': True},
                    {'id': 'llm-pi-ai', 'disabled': False, 'config': {'providers': {
                        'compact-test': {'api': 'openai-completions', 'apiKeyEnv': 'RSI_MOCK_API_KEY',
                                         'baseURL': f'http://127.0.0.1:{server.server_port}/v1',
                                         'compat': {'thinkingFormat': 'deepseek'},
                                         'models': [{'id': 'compact-test', 'name': 'Mock',
                                                     'contextWindow': 1048576, 'maxTokens': 262144,
                                                     'reasoningEfforts': {'high': 'high', 'max': 'max'}}]}}}},
                    {'id': 'agent-default-model', 'config': {
                        'provider': 'compact-test', 'model': 'compact-test', 'reasoningEffort': 'max'}},
                    {'id': 'session-persistence-jsonl', 'config': {
                        'root': str(root / 'sessions'), 'compression': 'none'}},
                ]))
                env = dict(os.environ, DSH_HOME=str(root / 'state'), RSI_MOCK_API_KEY='mock-only')
                argv = [os.environ['DSH_BIN'], '--profile', 'headless',
                        '--patch', str(model_patch), '--patch', str(PATCH), '--json']

                def invoke(prompt, session=None):
                    command = argv + ([] if session is None else ['--session-id', session]) + ['-']
                    result = subprocess.run(command, input=prompt, text=True, cwd=root,
                                            env=env, capture_output=True, timeout=60)
                    diagnostics.append(result.stderr[-2500:])
                    self.assertEqual(result.returncode, 0, result.stderr[-2500:] + result.stdout[-1500:])
                    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]

                first = invoke(f'Retain {marker}.\n' + 'old context ' * 180000)
                opening = next(event for event in first if event.get('type') == 'session')
                session = opening.get('sessionId') or opening.get('id')
                self.assertIsInstance(session, str, opening)
                invoke('Continue the same task.\n' + 'recent context ' * 90000, session)
                invoke('Continue from the retained state.', session)
                summaries = [i for i, r in enumerate(requests)
                             if 'You are now acting as a compaction engine' in json.dumps(r['messages'])]
                self.assertTrue(summaries, f'no automatic summary: {diagnostics}; '
                                f'request sizes: {[len(json.dumps(r["messages"])) for r in requests]}')
                index = summaries[-1]
                self.assertLess(index, len(requests) - 1, 'no task request after compaction')
                resumed = json.dumps(requests[index + 1]['messages'])
                self.assertIn('<compacted-summary>', resumed)
                self.assertIn(marker, resumed)
                self.assertNotIn('old context ' * 100, resumed)
                self.assertLess(len(resumed), len(json.dumps(requests[index]['messages'])))
                for i, request in enumerate(requests):
                    cap = request.get('max_tokens', request.get('max_completion_tokens'))
                    if i in summaries:
                        self.assertEqual(cap, 8192)
                    else:
                        self.assertGreater(cap, 0)
                        self.assertLessEqual(cap, 262144)
                        self.assertEqual(request.get('reasoning_effort'), 'max')
                self.assertEqual(requests[0].get('max_tokens', requests[0].get('max_completion_tokens')), 262144)
                following = requests[index + 1]
                self.assertEqual(following.get('max_tokens', following.get('max_completion_tokens')), 262144)
                logs = list((root / 'sessions').rglob('*.jsonl'))
                durable = '\n'.join(path.read_text() for path in logs)
                self.assertIn('compaction/summary', durable)
                self.assertIn('compaction/end', durable)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    @unittest.skipUnless(os.environ.get('DSH_BIN'), 'optional DSH integration')
    def test_research_adapter_keeps_native_history_across_round_directories(self):
        from research.memory import read, write
        from research.research import Research
        import sys
        requests = []
        marker = 'RSI_native_persistence_8297'

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                event = {'id':'mock', 'object':'chat.completion.chunk', 'created':0,
                    'model':'persist-test', 'choices':[{'index':0,
                        'delta':{'role':'assistant','content':'acknowledged'}, 'finish_reason':None}]}
                self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
                event['choices'] = [{'index':0,'delta':{},'finish_reason':'stop'}]
                self.wfile.write(('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode())

        server = ThreadingHTTPServer(('127.0.0.1',0), Provider)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix='rsi-persist-') as tmp:
                root = Path(tmp)
                patch = root/'model.json'
                write(patch, [
                    {'id':'llm-deepseek', 'disabled':True},
                    {'id':'llm-pi-ai', 'disabled':False, 'config':{'providers':{
                        'persist-test':{'api':'openai-completions', 'apiKeyEnv':'RSI_MOCK_API_KEY',
                            'baseURL':f'http://127.0.0.1:{server.server_port}/v1',
                            'models':[{'id':'persist-test','name':'Mock','contextWindow':65536,'maxTokens':1024}]}}}},
                    {'id':'agent-default-model','config':{'provider':'persist-test','model':'persist-test'}},
                ])
                research = Research(root/'research', {'model':'mock-only'})
                research.sync([])
                env = dict(os.environ, DSH_HOME=str(root/'state'), RSI_MOCK_API_KEY='mock-only')
                sessions = []
                for number in (1,2):
                    run = root/f'candidate-{number}'
                    run.mkdir()
                    prompt = root/f'prompt-{number}.txt'
                    prompt.write_text('Remember '+marker if number == 1 else 'Recall the previous marker.')
                    agent = dict(argv=[os.environ['DSH_BIN'],'--profile','headless','--patch',str(patch),
                        '--patch',str(PATCH),'--json','{prompt}'], usage_format='dsh-jsonl',
                        persistence={'kind':'dsh'}, research=dict(root=str(research.root),
                            binding={'model':'mock-only'},turn=f'round-{number:03d}'))
                    config = root/f'agent-{number}.json'
                    write(config, agent)
                    command = [sys.executable,'-B','-m','research.harness',
                        str(config), str(prompt), str(run),'--cwd-run']
                    output = subprocess.run(command,cwd=PATCH.parents[1],env=env,
                        capture_output=True,timeout=60)
                    self.assertEqual(output.returncode,0,output.stderr.decode()[-2500:] + output.stdout.decode()[-3500:])
                    sessions.append(read(research.private/'cursor.json')['session_id'])
                    # Replaying an already completed turn performs no HTTP request.
                    before = len(requests)
                    replay = subprocess.run(command,cwd=PATCH.parents[1],env=env,
                        capture_output=True,timeout=60)
                    self.assertEqual(replay.returncode,0,replay.stderr.decode()[-1500:])
                    self.assertEqual(output.stdout,replay.stdout)
                    self.assertEqual(len(requests),before)
                self.assertEqual(sessions[0],sessions[1])
                task_requests = [r for r in requests if not str(r['messages'][0].get('content','')).startswith(
                    'Create a concise title for an AI coding-assistant session')]
                self.assertEqual(len(task_requests),2)
                self.assertIn(marker,json.dumps(task_requests[1]['messages']))
                self.assertTrue(list((research.private/'native').rglob('*.jsonl')))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
