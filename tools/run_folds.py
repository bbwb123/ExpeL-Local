"""Sequential independent folds; only writes aggregate after every fold completes."""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expel.common import load, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3.5:9b')
    p.add_argument('--out', default='runs/folds')
    p.add_argument('--environment', choices=['local','wikipedia','webshop'], default='local')
    p.add_argument('--folds', type=int, default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--webshop-url', default='http://127.0.0.1:3000')
    p.add_argument('--webshop-timeout', type=int, default=120)
    p.add_argument('--webshop-catalog', choices=['unknown','small','full'], default='unknown')
    p.add_argument('--demos', type=int, default=None)
    p.add_argument('--retriever', choices=['bm25','mpnet'], default='bm25')
    p.add_argument('--num-ctx', type=int, default=32768)
    p.add_argument('--success-batch', type=int, default=8)
    a = p.parse_args()
    webshop = a.environment == 'webshop'
    folds = a.folds if a.folds is not None else (2 if webshop else 4)
    demos = a.demos if a.demos is not None else (2 if webshop else 6)
    if folds < 2 or not 1 <= demos <= (2 if webshop else 6):
        p.error('At least two folds and a valid demonstration count are required.')
    destination = Path(a.out).resolve()
    all_summaries = []
    identity = None
    for fold in range(folds):
        out = destination / f'fold-{fold}'
        command = [sys.executable, '-m', 'expel', 'pipeline', '--out', str(out), '--model', a.model,
                        '--environment', a.environment, '--retriever', a.retriever,
                        '--folds',str(folds),'--fold', str(fold), '--train-limit','0','--eval-limit','0',
                        '--retries','3','--demos',str(demos),'--top-k',str(demos),'--success-batch',str(a.success_batch),
                        '--base-url',a.base_url,'--webshop-url',a.webshop_url,
                        '--webshop-timeout',str(a.webshop_timeout),'--webshop-catalog',a.webshop_catalog,
                        '--num-ctx',str(a.num_ctx)]
        if a.data:
            command += ['--data',str(Path(a.data).resolve())]
        subprocess.run(command, cwd=ROOT, check=True)
        manifest = load(out/'manifest.json')
        run_identity = {k:manifest[k] for k in ('model','dataset_sha256','implementation_sha256')}
        if identity is not None and run_identity != identity:
            raise RuntimeError('Model/data/code differs across folds; aggregation rejected.')
        identity = run_identity
        all_summaries.append(load(out/'summary.json'))
    aggregated = []
    for i, condition in enumerate(all_summaries[0]['conditions']):
        row = {'condition':condition['condition'],'folds':folds}
        metrics = ('success_rate','mean_reward') if webshop else ('em','f1')
        for metric in (*metrics,'mean_steps','mean_seconds'):
            values = [s['conditions'][i][metric] for s in all_summaries]
            row[metric+'_mean'] = statistics.mean(values)
            row[metric+'_se'] = statistics.stdev(values)/math.sqrt(len(values))
        aggregated.append(row)
    save(destination/'folds_summary.json',aggregated)
    with (destination/'folds_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(aggregated[0]))
        writer.writeheader(); writer.writerows(aggregated)
    print(json.dumps(aggregated,indent=2))


if __name__ == '__main__': main()
