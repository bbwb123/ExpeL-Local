import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print('task:', d['question'])
print('status:', d['status'], '| metrics:', d['metrics'], '| bought:', d['prediction'], d['selected_options'])
for s in d['steps']:
    out = s['observation'].splitlines()[0][:80] if s['observation'] else ''
    print(f"{s['index']:>2}. {s['action']}[{s['argument'][:70]}] -> {s['page_type']} | {out}")