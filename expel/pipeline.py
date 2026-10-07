import csv
import json
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import as_demo, reflect, run_episode
from .common import fingerprint, load, normalize, save, succeeded
from .environment import LiveWiki, LocalWiki, corpus_from_rows
from .memory import RuleBook, extract_update, extraction_groups
from .retrieval import ExperienceRetriever

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('react', 'insights_only', 'retrieval_only', 'expel')


def demo_question(demo):
    match = re.search(r'(?:Question|Instruction):\s*([^\n]+)', demo)
    if not match:
        raise ValueError('Demonstration has no Question or Instruction.')
    return match.group(1)


def split_rows(rows, demonstrations, seed=42, folds=4, fold=0, train_limit=8, eval_limit=4):
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError('Require folds >= 2 and 0 <= fold < folds.')
    demo_questions = {normalize(demo_question(d)) for d in demonstrations}
    seen_ids, seen_questions, clean = set(), set(), []
    for row in rows:
        key = normalize(row['question'])
        if row['id'] in seen_ids or key in seen_questions or key in demo_questions:
            continue
        seen_ids.add(row['id'])
        seen_questions.add(key)
        clean.append(row)
    random.Random(seed).shuffle(clean)
    train = [r for i, r in enumerate(clean) if i % folds != fold]
    valid = [r for i, r in enumerate(clean) if i % folds == fold]
    train = train[:train_limit] if train_limit else train
    valid = valid[:eval_limit] if eval_limit else valid
    if not train or not valid:
        raise ValueError('Both experience and validation splits must be nonempty.')
    return train, valid


class Experiment:
    def __init__(self, config, model):
        self.config, self.model = config, model
        self.out = Path(config['out'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.is_webshop = config['environment'] == 'webshop'
        self.run_episode, self.reflect, self.as_demo = run_episode, reflect, as_demo
        if self.is_webshop:
            from .webshop import WebShop, run_webshop_episode, reflect_webshop, as_webshop_demo
            self.run_episode, self.reflect, self.as_demo = run_webshop_episode, reflect_webshop, as_webshop_demo
            self.environment = WebShop(config['webshop_url'], config['webshop_timeout'])
        rows = load(config['data'])
        if self.is_webshop:
            # Never carry upstream key/target product labels into experiments.
            from .webshop import clean_instruction
            rows = [{'id': r.get('id', r['session_idx']), 'session_idx': r['session_idx'],
                     'question': clean_instruction(r.get('question', r.get('task', '')))} for r in rows]
            if any(not r['question'] or not re.fullmatch(r'fixed_\d+', r['session_idx']) for r in rows):
                raise ValueError('WebShop data must have a public instruction and fixed_<index> session_idx.')
            if len({r['session_idx'] for r in rows}) != len(rows):
                raise ValueError('WebShop task session indices must be unique.')
        demos = load(ROOT / ('data/webshop_demonstrations.json' if self.is_webshop else 'data/paper_demonstrations.json'))
        if config['demos'] > len(demos):
            raise ValueError(f'This benchmark provides only {len(demos)} fixed demonstrations.')
        self.demos = demos[:config['demos']]
        self.train, self.valid = split_rows(rows, demos, config['seed'], config['folds'], config['fold'],
                                           config['train_limit'], config['eval_limit'])
        if not self.is_webshop:
            self.environment = (LocalWiki(corpus_from_rows(rows)) if config['environment'] == 'local'
                                else LiveWiki(self.out / 'wiki_cache'))
        self.identity = model.identity()
        self.split = {'experience_ids': [r['id'] for r in self.train],
                      'validation_ids': [r['id'] for r in self.valid]}
        manifest = {'config': config, 'dataset_sha256': fingerprint(rows),
                    'demonstrations_sha256': fingerprint(demos), 'split': self.split,
                    'model': self.identity, 'sampling': {'think': getattr(model, 'think', False), 'think_scope': 'agent',
                    'temperature': getattr(model, 'temperature', 0.0), 'num_predict': getattr(model, 'num_predict', None)},
                    'protocol': 'expel-webshop-http-v1' if self.is_webshop else 'expel-local-v1',
                    'execution_kind': getattr(model, 'execution_kind', 'scripted-test'),
                    'parameter_updates': False, 'implementation_sha256': fingerprint(
                        {p.name: p.read_text(encoding='utf-8') for p in sorted((ROOT / 'expel').glob('*.py'))})}
        path = self.out / 'manifest.json'
        if path.exists():
            old = load(path)
            if old['experiment_fingerprint'] != fingerprint(manifest):
                raise RuntimeError('Run directory belongs to different configuration/model/data/code. Choose a new --out directory.')
        else:
            save(path, {'experiment_fingerprint': fingerprint(manifest), **manifest,
                        'created_at_utc': datetime.now(timezone.utc).isoformat(),
                        'runtime': {'python': sys.version, 'platform': platform.platform()}})

    def preflight(self):
        if self.is_webshop:
            # Validate goal mapping before expensive model calls in each stage/resume.
            self.environment.reset(self.train[0])

    def check_model(self):
        current = self.model.identity()
        if current['digest'] != self.identity['digest']:
            raise RuntimeError('Model digest changed during experiment. Do not mix model versions.')

    def read_collected(self):
        path = self.out / 'experience.json'
        if not path.exists():
            raise RuntimeError('Run collect first.')
        records = load(path)
        if [r['id'] for r in records] != self.split['experience_ids']:
            raise RuntimeError('Experience stage incomplete. Resume collect first.')
        return records

    def collect(self):
        self.preflight()
        path = self.out / 'experience.json'
        records = load(path) if path.exists() else []
        if [r['id'] for r in records] != self.split['experience_ids'][:len(records)]:
            raise RuntimeError('Experience checkpoint is not a prefix of this split.')
        for index, row in enumerate(self.train[len(records):], start=len(records)):
            self.check_model()
            self.model.calls = []
            episodes, reflections = [], []
            for attempt in range(self.config['retries'] + 1):
                episode = self.run_episode(self.model, self.environment, row, self.demos,
                            reflections=reflections, max_steps=self.config['max_steps'])
                episode['attempt'] = attempt
                episodes.append(episode)
                label = ('success=' + str(int(episode['metrics']['success'])) +
                         f' reward={episode["metrics"]["reward"]:.3f}' if self.is_webshop else
                         f'EM={episode["metrics"]["em"]:.0f}')
                print(f'collect {index+1}/{len(self.train)} attempt={attempt} {label}', flush=True)
                if succeeded(episode):
                    break
                if attempt < self.config['retries']:
                    reflections.append(self.reflect(self.model, episode))
            self.check_model()
            records.append({'id': row['id'], 'episodes': episodes, 'all_calls': self.model.calls})
            save(path, records)
        print('Experience saved:', path, flush=True)

    def extract(self):
        records = self.read_collected()
        groups = extraction_groups(records, self.config['success_batch'])
        path = self.out / 'insights.json'
        source_hash = fingerprint(records)
        state = load(path) if path.exists() else {'source_hash': source_hash, 'rules': [], 'updates': [], 'next_group': 0}
        if state['source_hash'] != source_hash:
            raise RuntimeError('Experience changed after extraction. Use a new run directory.')
        book = RuleBook(state['rules'], self.config['max_rules'])
        for index, group in enumerate(groups[state['next_group']:], start=state['next_group']):
            self.check_model()
            self.model.calls = []
            update = extract_update(self.model, book, group)
            state['updates'].append({'index': index, 'kind': group['kind'],
                       'source_ids': [e['id'] for e in group['episodes']], **update,
                       'calls': self.model.calls})
            state.update(rules=book.rules, next_group=index+1)
            save(path, state)
            print(f'extract {index+1}/{len(groups)} rules={len(book.rules)} {update["status"]}', flush=True)
        self.check_model()
        state.update(complete=True, contrast_groups=sum(g['kind'] == 'contrast' for g in groups),
                     success_groups=sum(g['kind'] == 'success_batch' for g in groups),
                     warning='' if book.rules else 'No rules extracted; this run has not demonstrated insight learning.')
        save(path, state)

    def evaluate(self):
        records = self.read_collected()
        state = load(self.out / 'insights.json')
        if not state.get('complete') or state['source_hash'] != fingerprint(records):
            raise RuntimeError('Complete extract before evaluation.')
        if not state['rules']:
            raise RuntimeError('No learned rules. Collect more successful experience in a new run before four-condition evaluation.')
        self.preflight()
        successful = [next(e for e in r['episodes'] if succeeded(e)) for r in records
                      if any(succeeded(e) for e in r['episodes'])]
        demo_episodes = []
        for i, demo in enumerate(self.demos):
            # Fixed, manually authored examples are also eligible for experience recall (paper Alg. 1).
            demo_episodes.append({'id': f'paper-demo-{i}', 'question': demo_question(demo),
                                  'demo_text': demo, 'steps': [], 'metrics': {'em': 1}})
        pool = successful + demo_episodes
        if set(e['id'] for e in pool) & set(self.split['validation_ids']):
            raise RuntimeError('Validation leakage detected.')
        retriever = ExperienceRetriever(pool, self.config['retriever'])
        memory_hash = fingerprint(state)
        experience_hash = fingerprint(records)
        self.model.calls = []
        path = self.out / 'evaluation.json'
        result = load(path) if path.exists() else {'memory_hash': memory_hash,
                      'experience_hash': experience_hash, 'rows': []}
        if result['memory_hash'] != memory_hash or result['experience_hash'] != experience_hash:
            raise RuntimeError('Frozen memory changed since evaluation started.')
        done = {(r['id'], r['condition']) for r in result['rows']}
        # Round-robin conditions limit temporal latency bias; each episode has fresh environment/state.
        for index, row in enumerate(self.valid):
            retrieved = retriever.retrieve(row['question'], self.config['top_k'])
            retrieved_demos = [e['demo_text'] if 'demo_text' in e else self.as_demo(e) for e in retrieved]
            for condition in CONDITIONS:
                if (row['id'], condition) in done:
                    continue
                self.check_model()
                rules = state['rules'] if condition in ('insights_only', 'expel') else []
                use_retrieval = condition in ('retrieval_only', 'expel')
                demos = retrieved_demos if use_retrieval else self.demos
                self.model.calls = []
                episode = self.run_episode(self.model, self.environment, row, demos, rules,
                                      max_steps=self.config['max_steps'])
                episode.update(condition=condition, reference=None if self.is_webshop else row['answer'],
                               retrieved_ids=[e['id'] for e in retrieved] if use_retrieval else [])
                self.check_model()
                result['rows'].append(episode)
                save(path, result)
                label = (f'success={episode["metrics"]["success"]:.0f} reward={episode["metrics"]["reward"]:.3f}'
                         if self.is_webshop else f'EM={episode["metrics"]["em"]:.0f}')
                print(f'eval {index+1}/{len(self.valid)} {condition}: {label}', flush=True)
        if fingerprint(load(self.out / 'insights.json')) != memory_hash:
            raise RuntimeError('Memory was modified during validation.')
        report(self.out)


def report(out):
    out = Path(out)
    data, manifest = load(out / 'evaluation.json'), load(out / 'manifest.json')
    expected_ids = set(manifest['split']['validation_ids'])
    is_webshop = manifest['config']['environment'] == 'webshop'
    summaries = []
    for condition in CONDITIONS:
        rows = [r for r in data['rows'] if r['condition'] == condition]
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise RuntimeError('Evaluation is incomplete or duplicated; resume evaluate before reporting.')
        n = len(rows)
        metrics = ({'success_rate': sum(r['metrics']['success'] for r in rows)/n,
                    'mean_reward': sum(r['metrics']['reward'] for r in rows)/n} if is_webshop else
                   {'em': sum(r['metrics']['em'] for r in rows)/n, 'f1': sum(r['metrics']['f1'] for r in rows)/n})
        summaries.append({'condition': condition, 'n': n, **metrics,
            'mean_steps': sum(len(r['steps']) for r in rows)/n,
            'mean_tool_calls': sum(sum(s['action'] in ('Search','Lookup','Click') for s in r['steps']) for r in rows)/n,
            'mean_seconds': sum(r['seconds'] for r in rows)/n,
            'prompt_tokens': sum(c['prompt_tokens'] for r in rows for c in r['calls']),
            'output_tokens': sum(c['output_tokens'] for r in rows for c in r['calls']),
            'format_errors': sum(sum(s['action']=='Invalid' for s in r['steps']) for r in rows),
            'max_steps_count': sum(r['status']=='max_steps' for r in rows),
            'context_warnings': sum(c.get('context_warning', False) for r in rows for c in r['calls'])})
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    collection = load(out / 'experience.json')
    insights = load(out / 'insights.json')
    prep_calls = [c for r in collection for c in r['all_calls']] + [c for u in insights['updates'] for c in u['calls']]
    prep = {'llm_calls': len(prep_calls), 'prompt_tokens': sum(c['prompt_tokens'] for c in prep_calls),
            'output_tokens': sum(c['output_tokens'] for c in prep_calls),
            'seconds': sum(c['seconds'] for c in prep_calls)}
    by = {c: {r['id']: r for r in data['rows'] if r['condition']==c} for c in CONDITIONS}
    paired = {'baseline_wrong_expel_correct': sum(not succeeded(by['react'][i]) and succeeded(by['expel'][i]) for i in expected_ids),
              'baseline_correct_expel_wrong': sum(succeeded(by['react'][i]) and not succeeded(by['expel'][i]) for i in expected_ids)}
    save(out / 'summary.json', {'conditions': summaries, 'preparation_cost': prep, 'paired': paired})
    text = '# ExpeL 本地实验结果\n\n'
    text += f"模型：{manifest['model']['name']}；环境：{manifest['config']['environment']}；检索：{manifest['config']['retriever']}。\n\n"
    if manifest['execution_kind'] == 'ollama':
        text += '这是当前配置下 Ollama 模型调用生成的结果；小样本结果不能等同论文完整复现。\n\n'
    else:
        text += '这是脚本模拟的流程测试，不能作为模型能力或 benchmark 成绩。\n\n'
    if is_webshop:
        text += f'WebShop 服务：{manifest["config"]["webshop_url"]}；商品库范围（用户声明）：{manifest["config"].get("webshop_catalog", "unknown")}。\n\n'
        text += '成功率按购买后环境奖励等于 1.0 计算；平均奖励保留部分完成的得分。未购买而耗尽步数按 0 分计算。\n\n'
        text += '| 条件 | 样本数 | 成功率 | 平均奖励 | 平均步数 | 平均秒数 |\n|---|---:|---:|---:|---:|---:|\n'
    else:
        text += '| 条件 | 样本数 | EM | F1 | 平均步数 | 平均秒数 |\n|---|---:|---:|---:|---:|---:|\n'
    for row in summaries:
        m1, m2 = ('success_rate', 'mean_reward') if is_webshop else ('em', 'f1')
        text += f"| {row['condition']} | {row['n']} | {row[m1]:.3f} | {row[m2]:.3f} | {row['mean_steps']:.2f} | {row['mean_seconds']:.1f} |\n"
    text += '\n经验收集与规则提炼的额外调用成本：\n\n```json\n' + json.dumps(prep, indent=2) + '\n```\n'
    text += '\n逐题对照：\n\n```json\n' + json.dumps(paired, indent=2) + '\n```\n'
    text += f"\n成功/失败对比组：{insights['contrast_groups']}；成功批次：{insights['success_groups']}；规则数：{len(insights['rules'])}。\n"
    if insights['contrast_groups'] == 0:
        text += '\n本次没有实际覆盖成功/失败配对提炼；需扩大经验集再验证该机制。\n'
    else:
        text += f'\n本次实际覆盖 {insights["contrast_groups"]} 个成功/失败对比组；仍需扩大样本验证规则收益。\n'
    (out / 'report.md').write_text(text, encoding='utf-8')
    print(text)
