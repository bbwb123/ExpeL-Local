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
    p.add_argument('command', choices=['doctor','one','collect','extract','evaluate','pipeline','report',
                                       'webshop-check','webshop-export'])
    p.add_argument('--model', default='qwen3.5:9b')
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--out', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--environment', choices=['local','wikipedia','webshop'], default='local')
    p.add_argument('--webshop-url', default='http://127.0.0.1:3000', help='Official WebShop Flask server root URL')
    p.add_argument('--webshop-timeout', type=int, default=120)
    p.add_argument('--webshop-catalog', choices=['unknown','small','full'], default='unknown',
                   help='User-declared catalogue size, recorded but not independently verified')
    p.add_argument('--export-path', default='data/webshop_server_tasks.json')
    p.add_argument('--task-count', type=int, default=100)
    p.add_argument('--task-start', type=int, default=0)
    p.add_argument('--retriever', choices=['bm25','mpnet'], default='bm25')
    p.add_argument('--train-limit', type=int, default=8)
    p.add_argument('--eval-limit', type=int, default=4)
    p.add_argument('--folds', type=int, default=None)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--retries', type=int, default=2, help='Additional attempts per experience task')
    p.add_argument('--demos', type=int, default=2)
    p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--max-rules', type=int, default=10)
    p.add_argument('--success-batch', type=int, default=2)
    p.add_argument('--num-ctx', type=int, default=16384)
    p.add_argument('--num-predict', type=int, default=1024)
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    if a.command.startswith('webshop-'):
        a.environment = 'webshop'
    webshop = a.environment == 'webshop'
    a.out = a.out or ('runs/webshop-pilot' if webshop else 'runs/pilot')
    a.data = a.data or str(ROOT / ('data/webshop_tasks.json' if webshop else 'data/hotpotqa_upstream_sample.json'))
    a.max_steps = a.max_steps if a.max_steps is not None else (15 if webshop else 7)
    a.folds = a.folds if a.folds is not None else (2 if webshop else 4)
    if any(getattr(a,k)<0 for k in ['train_limit','eval_limit','retries']):
        p.error('Limits/retries must not be negative.')
    if not 1 <= a.demos <= (2 if webshop else 6) or a.top_k != a.demos:
        p.error('Use 1..2 demonstrations for WebShop (1..6 for QA) and equal --top-k.')
    if min(a.max_steps,a.max_rules,a.success_batch,a.num_ctx,a.num_predict,a.timeout,a.webshop_timeout,a.task_count) < 1 or a.task_start < 0:
        p.error('Step, context and generation limits must be positive.')
    if a.command == 'report':
        report(a.out)
        return
    if a.command in ('webshop-check','webshop-export'):
        from .webshop import WebShop, clean_instruction
        env = WebShop(a.webshop_url, a.webshop_timeout)
        if a.command == 'webshop-check':
            row = load(a.data)[0]
            observation = env.reset({'session_idx':row['session_idx'],
                                     'question':clean_instruction(row.get('question',row.get('task','')))})
            result = {'server':a.webshop_url, 'session_idx':row['session_idx'],
                      'instruction':env.instruction, 'observation':observation,
                      'note':'Instruction mapping verified; no product purchase or model accuracy measured.'}
            save(Path(a.out)/'webshop_connectivity.json', result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            destination = Path(a.export_path)
            if destination.exists():
                raise FileExistsError('Export destination already exists. Use a new --export-path.')
            rows = []
            for i in range(a.task_start, a.task_start + a.task_count):
                session_idx = f'fixed_{i}'
                env.reset({'session_idx':session_idx, 'question':''})
                rows.append({'id':session_idx,'session_idx':session_idx,'question':env.instruction})
                print(f'export {len(rows)}/{a.task_count}', flush=True)
            save(destination, rows)
            save(destination.with_suffix('.source.json'), {'server':a.webshop_url, 'task_count':len(rows),
                 'task_start':a.task_start, 'catalogue_user_declared':a.webshop_catalog,
                 'note':'Public instructions fetched from deterministic server goals; no target-product fields included.'})
            print('Task file saved:', destination)
        return
    model = Ollama(a.model, a.base_url, a.num_ctx, a.num_predict, a.timeout, a.seed)
    if a.command == 'doctor':
        info = model.identity()
        reply = model.chat([{'role':'user','content':'Reply exactly: LOCAL_MODEL_OK'}], 'connectivity')
        result = {'model': info, 'reply': reply, 'calls': model.calls,
                  'note': 'This checks model/API connectivity only, not benchmark accuracy.'}
        if webshop:
            from .webshop import WebShop, clean_instruction
            row = load(a.data)[0]
            env = WebShop(a.webshop_url, a.webshop_timeout)
            env.reset({'session_idx':row['session_idx'],
                       'question':clean_instruction(row.get('question',row.get('task','')))})
            result['webshop'] = {'url':a.webshop_url,'instruction':env.instruction,'task_mapping_verified':True}
        save(Path(a.out)/'connectivity.json',result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == 'one':
        # Probe uses a training-side task, never a selected validation task.
        exp = Experiment({k:v for k,v in vars(a).items() if k != 'command'}, model)
        model.calls = []
        result = exp.run_episode(model, exp.environment, exp.train[0], exp.demos, max_steps=a.max_steps)
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
