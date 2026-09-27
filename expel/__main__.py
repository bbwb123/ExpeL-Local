import argparse
import json
import sys
from pathlib import Path

from .agent import run_episode
from .client import Ollama
from .common import load, save
from .environment import LocalWiki, corpus_from_rows
from .pipeline import Experiment, ROOT, report


def main():
    p = argparse.ArgumentParser(description='ExpeL local inference-only reproduction')
    p.add_argument('command', choices=['doctor','one','collect','extract','evaluate','pipeline','report'])
    p.add_argument('--model', default='qwen3.5:9b')
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--out', default='runs/pilot')
    p.add_argument('--data', default=str(ROOT/'data/hotpotqa_upstream_sample.json'))
    p.add_argument('--environment', choices=['local','wikipedia'], default='local')
    p.add_argument('--retriever', choices=['bm25','mpnet'], default='bm25')
    p.add_argument('--train-limit', type=int, default=8)
    p.add_argument('--eval-limit', type=int, default=4)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=7)
    p.add_argument('--retries', type=int, default=2, help='Additional attempts per experience task')
    p.add_argument('--demos', type=int, default=2)
    p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--max-rules', type=int, default=10)
    p.add_argument('--success-batch', type=int, default=2)
    p.add_argument('--num-ctx', type=int, default=16384)
    p.add_argument('--num-predict', type=int, default=1024)
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    if any(getattr(a,k)<0 for k in ['train_limit','eval_limit','retries']):
        p.error('Limits/retries must not be negative.')
    if not 1 <= a.demos <= 6 or a.top_k != a.demos:
        p.error('Use 1..6 demonstrations and equal --top-k for comparable example counts.')
    if min(a.max_steps,a.max_rules,a.success_batch,a.num_ctx,a.num_predict,a.timeout) < 1:
        p.error('Step, context and generation limits must be positive.')
    if a.command == 'report':
        report(a.out)
        return
    model = Ollama(a.model, a.base_url, a.num_ctx, a.num_predict, a.timeout, a.seed)
    if a.command == 'doctor':
        info = model.identity()
        reply = model.chat([{'role':'user','content':'Reply exactly: LOCAL_MODEL_OK'}], 'connectivity')
        result = {'model': info, 'reply': reply, 'calls': model.calls,
                  'note': 'This checks model/API connectivity only, not benchmark accuracy.'}
        save(Path(a.out)/'connectivity.json',result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == 'one':
        # Probe uses a training-side task, never a selected validation task.
        exp = Experiment({k:v for k,v in vars(a).items() if k != 'command'}, model)
        model.calls = []
        result = run_episode(model, exp.environment, exp.train[0], exp.demos, max_steps=a.max_steps)
        save(Path(a.out)/'single_example.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'calls'}, ensure_ascii=False, indent=2))
        return
    exp = Experiment({k:v for k,v in vars(a).items() if k != 'command'}, model)
    if a.command == 'pipeline':
        exp.collect()
        exp.extract()
        exp.evaluate()
    else:
        getattr(exp, a.command)()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, ImportError, OSError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
