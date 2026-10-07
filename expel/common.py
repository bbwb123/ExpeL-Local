import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def scores(prediction, reference):
    p, g = normalize(prediction), normalize(reference)
    em = float(p == g)
    # HotpotQA's special handling for categorical answers matters.
    if (p in ('yes', 'no', 'noanswer') or g in ('yes', 'no', 'noanswer')) and p != g:
        return {'em': em, 'f1': 0.0}
    overlap = sum((Counter(p.split()) & Counter(g.split())).values())
    f1 = 2 * overlap / (len(p.split()) + len(g.split())) if overlap else em
    return {'em': em, 'f1': f1}


def trajectory(episode):
    return '\n'.join('Model: ' + s['output'] + '\nObservation: ' + s['observation']
                     for s in episode['steps'])


def succeeded(episode):
    metrics = episode['metrics']
    return metrics.get('success', metrics.get('em', 0)) == 1
