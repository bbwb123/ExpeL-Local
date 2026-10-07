import copy
import os
import json
import time
import urllib.error
import urllib.request


class TruncatedOutput(RuntimeError):
    """Generation limit reached before any visible content (e.g. runaway thinking)."""


class Ollama:
    """Native local API; no keys, training, adapters, or remote-model fallback."""
    execution_kind = 'ollama'
    def __init__(self, model='qwen3.5:9b', base_url='http://127.0.0.1:11434',
                 num_ctx=16384, num_predict=1024, timeout=300, seed=42):
        self.model, self.base_url = model, base_url.rstrip('/')
        self.num_ctx, self.num_predict = num_ctx, num_predict
        self.timeout, self.seed = timeout, seed
        # Sampling knobs come from EXPEL_THINK / EXPEL_TEMPERATURE and are recorded in manifest.json.
        self.think = os.environ.get('EXPEL_THINK', '0') == '1'
        self.temperature = float(os.environ.get('EXPEL_TEMPERATURE', '0'))
        self.calls = []

    def request(self, route, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + route, data=data,
                                        headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Ollama HTTP {exc.code}: {exc.read().decode(errors="replace")[:600]}') from exc
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(f'Cannot reach Ollama at {self.base_url}. Start Ollama; check model and timeout. {exc}') from exc

    def identity(self):
        models = self.request('/api/tags')['models']
        selected = next((m for m in models if m['name'] == self.model), None)
        if selected is None:
            raise RuntimeError(f'Model {self.model!r} missing. Run: ollama pull {self.model}')
        return {'name': selected['name'], 'digest': selected['digest'],
                'details': selected.get('details', {}),
                'ollama_version': self.request('/api/version').get('version')}

    def chat(self, messages, purpose='agent'):
        # Conservative preflight, not an exact token counter. Do not silently drop history.
        if sum(len(m['content']) for m in messages) > self.num_ctx * 2:
            raise RuntimeError('Prompt exceeds conservative context budget. Use a new run with more --num-ctx, '
                               'or fewer --demos/--top-k/--success-batch. No content was silently truncated.')
        # Thinking only for the acting policy (baseline and ExpeL alike). Reflection/insight extraction
        # keep think=false: their long thinking can exhaust num_predict and leave content empty.
        think = self.think and purpose == 'agent'
        body = {'model': self.model, 'messages': messages, 'stream': False, 'think': think,
                'keep_alive': '10m', 'options': {'temperature': self.temperature, 'seed': self.seed,
                'num_ctx': self.num_ctx, 'num_predict': self.num_predict}}
        start = time.monotonic()
        response = self.request('/api/chat', body)
        text = response.get('message', {}).get('content', '').strip()
        record = {'purpose': purpose, 'think': think, 'messages': copy.deepcopy(messages), 'output': text,
                  'prompt_tokens': response.get('prompt_eval_count', 0),
                  'output_tokens': response.get('eval_count', 0),
                  'seconds': time.monotonic() - start,
                  'done_reason': response.get('done_reason'),
                  'context_warning': response.get('prompt_eval_count', 0) +
                                     response.get('eval_count', 0) >= self.num_ctx - 64}
        self.calls.append(record)
        if not text:
            raise (TruncatedOutput if response.get('done_reason') == 'length' else RuntimeError)(f'Ollama returned empty content (purpose={purpose}, think={think}, '
                               f'done_reason={response.get("done_reason")}). Raise --num-predict or disable thinking.')
        return text
