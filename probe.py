"""Run the baseline WebShop actor on the experience-side tasks and print a one-line summary per task.
Usage: python probe.py <out.json> [n_tasks]   (respects EXPEL_THINK / EXPEL_TEMPERATURE / EXPEL_NUM_PREDICT)"""
import json, os, sys
from expel.client import Ollama
from expel.common import load
from expel.pipeline import ROOT, split_rows
from expel import webshop
from expel.webshop import WebShop, run_webshop_episode

if os.environ.get('EXPEL_PROMPT_FROM'):  # A/B test: take INSTRUCTIONS from another (backup) copy of webshop.py
    src = open(os.environ['EXPEL_PROMPT_FROM'], encoding='utf-8').read()
    webshop.INSTRUCTIONS = src.split("INSTRUCTIONS = '''", 1)[1].split("'''", 1)[0]

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
demos = load(ROOT / 'data/webshop_demonstrations.json')[:2]
train, _ = split_rows(load(ROOT / 'data/webshop_small_synth.json'), demos, 42, 2, 0, 8, 4)
model = Ollama(num_predict=int(os.environ.get('EXPEL_NUM_PREDICT', '1024')))
env, rows = WebShop(), []
for row in train[:n]:
    ep = run_webshop_episode(model, env, row, demos)
    acts = ' '.join(s['action'][0] + ('!' if s['observation'].startswith('Invalid') else '') for s in ep['steps'])
    print(f"{row['id']:>9} reward={ep['metrics']['reward']:.2f} steps={len(ep['steps']):>2} "
          f"{ep['seconds']:5.0f}s  {acts}", flush=True)
    rows.append({k: v for k, v in ep.items() if k not in ('calls', 'environment_requests')})
json.dump(rows, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
r = [x['metrics']['reward'] for x in rows]
print(f'mean_reward={sum(r)/len(r):.3f} success={sum(x["metrics"]["success"] for x in rows)}/{len(rows)} '
      f'bought={sum(x["status"]=="finished" for x in rows)}/{len(rows)}')
