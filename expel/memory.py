import json
import re

from .agent import as_demo


class RuleBook:
    """Paper scoring: ADD=2; EDIT/UPVOTE +1; DOWNVOTE -1; delete at zero."""
    def __init__(self, rules=None, max_rules=10):
        self.rules = list(rules or [])
        self.max_rules = max_rules
        self.next_id = max([r['id'] for r in self.rules], default=0) + 1

    def update(self, output):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', output.strip())
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError('Expected a JSON object.')
        operations = obj['operations']
        if not isinstance(operations, list) or len(operations) > 4:
            raise ValueError('Expected at most four operations.')
        # Validate the whole batch first, avoiding partially applied invalid responses.
        touched, known = set(), {r['id'] for r in self.rules}
        for op in operations:
            if not isinstance(op, dict) or not isinstance(op.get('op'), str):
                raise ValueError('Each operation must be an object with a string op.')
            name = op.get('op', '').upper()
            if name not in ('ADD', 'EDIT', 'UPVOTE', 'DOWNVOTE', 'AGREE', 'REMOVE'):
                raise ValueError('Unknown operation ' + name)
            if name in ('ADD', 'EDIT') and (not isinstance(op.get('text'), str) or not 1 <= len(op['text'].strip()) <= 500):
                raise ValueError('Rule text must contain 1..500 characters.')
            if name != 'ADD':
                if op.get('id') not in known or op['id'] in touched:
                    raise ValueError('Unknown or repeated rule id.')
                touched.add(op['id'])
        for op in operations:
            name = op['op'].upper()
            if name == 'ADD':
                if any(r['text'].casefold() == op['text'].strip().casefold() for r in self.rules):
                    continue
                self.rules.append({'id': self.next_id, 'text': op['text'].strip(), 'votes': 2})
                self.next_id += 1
            else:
                rule = next(r for r in self.rules if r['id'] == op['id'])
                rule['votes'] += -1 if name in ('DOWNVOTE', 'REMOVE') else 1
                if name == 'EDIT':
                    rule['text'] = op['text'].strip()
        self.rules = sorted([r for r in self.rules if r['votes'] > 0], key=lambda r: (-r['votes'], r['id']))[:self.max_rules]
        return operations


def extraction_groups(records, success_batch=8):
    groups, successes = [], []
    for record in records:
        good = [e for e in record['episodes'] if e['metrics']['em'] == 1]
        if not good:
            continue
        winner = good[0]
        successes.append(winner)
        for failed in record['episodes']:
            if failed['metrics']['em'] == 0:
                groups.append({'kind': 'contrast', 'episodes': [failed, winner]})
    for i in range(0, len(successes), success_batch):
        groups.append({'kind': 'success_batch', 'episodes': successes[i:i+success_batch]})
    return groups


def extract_update(model, book, group):
    instructions = '''You manage transferable procedural insights for a question-answering agent.
Compare FAILED and SUCCESSFUL attempts of the SAME task, or find common useful
practices across the provided successful tasks. Update existing general rules.
Do not store entity-specific facts, names, answers, task IDs, or verbatim solutions.
Return only JSON: {"operations": [{"op":"ADD","text":"..."}]}.
Other operations: {"op":"EDIT","id":1,"text":"..."},
{"op":"UPVOTE","id":1}, {"op":"DOWNVOTE","id":1}.
Use at most 4 operations, each existing id at most once. Empty operations is allowed.
Rules must be concise, under 500 characters. Prefer editing or downvoting redundant rules.
'''
    evidence = '\n\n'.join(('SUCCESSFUL' if e['metrics']['em'] else 'FAILED') + '\n' + as_demo(e)
                            for e in group['episodes'])
    messages = [{'role': 'system', 'content': instructions}, {'role': 'user', 'content':
                'Existing rules:\n' + json.dumps(book.rules, ensure_ascii=False) + '\nEvidence:\n' + evidence}]
    for attempt in range(2):
        output = model.chat(messages, 'insight_extraction')
        try:
            return {'operations': book.update(output), 'status': 'applied'}
        except (ValueError, KeyError, TypeError) as exc:
            messages.extend([{'role': 'assistant', 'content': output},
                             {'role': 'user', 'content': 'Invalid update: ' + str(exc) + '. Return corrected JSON.'}])
    return {'operations': [], 'status': 'parse_failed'}
