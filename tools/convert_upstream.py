"""One-time conversion of the pinned, trusted upstream sample to portable JSON.
Requires joblib/pandas; NOT needed to run the delivered project.
Never use this converter on untrusted pickle/joblib files.
"""
import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import joblib
import pandas as pd

source = Path(sys.argv[1])
destination = Path(__file__).resolve().parents[1] / 'data'
destination.mkdir(exist_ok=True)
# The authors serialized pandas 1.x Int64Index, removed in pandas 2.x.
compat = types.ModuleType('pandas.core.indexes.numeric')
for name in ('Int64Index', 'UInt64Index', 'Float64Index'):
    setattr(compat, name, pd.Index)
sys.modules.setdefault('pandas.core.indexes.numeric', compat)
frame = joblib.load(source / 'data/hotpotqa/hotpot-qa-distractor-sample.joblib')
rows = json.loads(frame.to_json(orient='records', force_ascii=False))
print('shape', frame.shape, 'columns', list(frame.columns))
out = destination / 'hotpotqa_upstream_sample.json'
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
# Extract literal demonstrations only; do not execute the upstream prompt module.
tree = ast.parse((source / 'prompts/hotpotQA.py').read_text(encoding='utf-8'))
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FEWSHOTS' for t in node.targets):
        demos = ast.literal_eval(node.value)
        (destination / 'paper_demonstrations.json').write_text(json.dumps(demos, ensure_ascii=False, indent=2), encoding='utf-8')
        print('demonstrations', len(demos))
        break
print('examples', len(rows), 'sha256', hashlib.sha256(out.read_bytes()).hexdigest())
print('first keys', list(rows[0]), 'context?', 'context' in rows[0])
