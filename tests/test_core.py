import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expel.agent import actor_messages, parse_action, run_episode
from expel.client import Ollama
from expel.common import load, save, scores
from expel.environment import LocalWiki, corpus_from_rows
from expel.memory import RuleBook, extraction_groups
from expel.pipeline import Experiment, ROOT, split_rows
from expel.retrieval import ExperienceRetriever


class ScriptModel:
    """Test fixture only. Never used in the CLI or benchmark reports."""
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def chat(self, messages, purpose='agent'):
        output = next(self.outputs)
        self.calls.append({'messages': copy.deepcopy(messages), 'output': output, 'purpose': purpose,
                           'prompt_tokens': 1, 'output_tokens': 1, 'seconds': 0.0})
        return output


class Tests(unittest.TestCase):
    def test_real_data_disjoint_and_no_demo_overlap(self):
        rows = load(ROOT/'data/hotpotqa_upstream_sample.json')
        demos = load(ROOT/'data/paper_demonstrations.json')
        all_valid = []
        for fold in range(4):
            train, valid = split_rows(rows, demos, folds=4, fold=fold, train_limit=0, eval_limit=0)
            self.assertFalse(set(r['id'] for r in train) & set(r['id'] for r in valid))
            all_valid.extend(r['id'] for r in valid)
        self.assertEqual(len(all_valid), len(set(all_valid)))
        self.assertGreater(len(all_valid), 90)

    def test_hotpot_categorical_scoring(self):
        self.assertEqual(scores('yes, correct', 'yes')['f1'], 0)
        self.assertEqual(scores('The United States.', 'United States')['em'], 1)
        self.assertAlmostEqual(scores('American film director', 'film director')['f1'], .8)

    def test_parser_rejects_multiple_actions_and_accepts_numbered(self):
        self.assertEqual(parse_action('Thought 1: next\nAction 1: Search[Somewhere]'),
                         ('Search','Somewhere','Thought: next\nAction: Search[Somewhere]'))
        with self.assertRaises(ValueError):
            parse_action('Action: Search[A]\nAction: Finish[B]')
        with self.assertRaises(ValueError):
            parse_action('Action: Finish[]')
        with self.assertRaises(ValueError):
            parse_action('Thought: guess\nAction: Search[A]\nObservation: invented')

    def test_environment_reset_and_lookup_cursor(self):
        env = LocalWiki({'Alpha': ['Founded in 1900.', 'Expanded in 1950.'], 'Beta':['A town.']})
        self.assertIn('No active', env.lookup('in'))
        env.search('Alpha')
        self.assertIn('1900', env.lookup('in'))
        self.assertIn('1950', env.lookup('in'))
        self.assertIn('No more', env.lookup('in'))
        env.search('Missing')
        self.assertIn('No active', env.lookup('in'))

    def test_tools_do_not_consume_labels(self):
        rows = [{'context': {'title':['Page'], 'sentences':[['Public paragraph.']]},
                 'answer': 'PRIVATE_ANSWER', 'supporting_facts': 'PRIVATE_LABEL'}]
        corpus = corpus_from_rows(rows)
        self.assertEqual(corpus, {'Page':['Public paragraph.']})

    def test_agent_uses_real_tool_observation_and_no_answer_leak(self):
        model = ScriptModel(['Action: Search[Alpha]', 'Action: Finish[PRIVATE_ANSWER]'])
        env = LocalWiki({'Alpha':['Evidence from environment.']})
        result = run_episode(model, env, {'id':'t','question':'Find Alpha?', 'answer':'PRIVATE_ANSWER'}, [])
        self.assertEqual(result['metrics']['em'], 1)
        for call in model.calls:
            self.assertNotIn('PRIVATE_ANSWER', json.dumps(call['messages']))
        self.assertIn('Evidence from environment', model.calls[1]['messages'][-1]['content'])

    def test_invalid_actions_exhaust_budget(self):
        m = ScriptModel(['bad', 'bad'])
        r = run_episode(m, LocalWiki({}), {'id':'t','question':'q','answer':'a'}, [], max_steps=2)
        self.assertEqual(r['status'], 'max_steps')
        self.assertEqual(len(r['steps']), 2)

    def test_rejected_transcript_is_not_reintroduced_into_context(self):
        fabricated = ('Thought 1: search\nAction 1: Search[Beer Wars]\n'
                      'Observation 1: FABRICATED_PRIVATE_FACT\n'
                      'Thought 2: done\nAction 2: Finish[wrong]')
        model = ScriptModel([fabricated, 'Action: Finish[Stone Brewing Co.]'])
        row = {'id':'t','question':'Which brewery?', 'answer':'Stone Brewing Co.'}
        result = run_episode(model, LocalWiki({}), row, [], max_steps=2)
        second_context = json.dumps(model.calls[1]['messages'])
        self.assertNotIn('FABRICATED_PRIVATE_FACT', second_context)
        self.assertNotIn('Finish[wrong]', second_context)
        self.assertEqual(result['metrics']['em'], 1)

    def test_rule_votes_and_atomic_validation(self):
        book = RuleBook()
        book.update('{"operations":[{"op":"ADD","text":"Check evidence."}]}')
        self.assertEqual(book.rules[0]['votes'], 2)
        before = copy.deepcopy(book.rules)
        with self.assertRaises(ValueError):
            book.update('{"operations":[{"op":"ADD","text":"Another rule."},{"op":"EDIT","id":99,"text":"bad"}]}')
        self.assertEqual(before, book.rules)
        book.update('{"operations":[{"op":"DOWNVOTE","id":1}]}')
        self.assertEqual(book.rules[0]['votes'], 1)
        book.update('{"operations":[{"op":"DOWNVOTE","id":1}]}')
        self.assertEqual(book.rules, [])

    def test_paired_and_success_batch_extraction(self):
        fail = {'id':'a','metrics':{'em':0}}
        good = {'id':'a','metrics':{'em':1}}
        groups = extraction_groups([{'episodes':[fail,good]}], 8)
        self.assertEqual([g['kind'] for g in groups], ['contrast','success_batch'])

    def test_retrieval_uses_question_similarity(self):
        pool = [{'id':'a','question':'Who directed the film?'}, {'id':'b','question':'Which river flows here?'}]
        result = ExperienceRetriever(pool).retrieve('Name the river',1)
        self.assertEqual(result[0]['id'], 'b')

    def test_native_ollama_payload_and_usage(self):
        model = Ollama()
        def request(route, body):
            self.assertEqual(route, '/api/chat')
            self.assertFalse(body['stream'])
            self.assertFalse(body['think'])
            self.assertEqual(body['model'], 'qwen3.5:9b')
            return {'message':{'content':'OK'},'prompt_eval_count':12,'eval_count':1}
        with patch.object(model, 'request', side_effect=request):
            self.assertEqual(model.chat([{'role':'user','content':'test'}]), 'OK')
        self.assertEqual(model.calls[0]['prompt_tokens'],12)

    def test_end_to_end_checkpoints_and_memory_freeze(self):
        # Scripted protocol test on synthetic fixtures. Not a benchmark result.
        rows = [{'id':str(i),'question':f'What is fixture {i}?','answer':f'VALUE_{i}',
                 'context':{'title':[f'P{i}'],'sentences':[[f'fixture {i} text.']]}} for i in range(6)]
        class FixtureModel:
            def __init__(self): self.calls=[]
            def identity(self): return {'name':'fixture','digest':'fixture-only'}
            def chat(self, messages, purpose='agent'):
                if purpose == 'reflection': output = 'Check the source before answering.'
                elif purpose == 'insight_extraction':
                    output = '{"operations":[{"op":"ADD","text":"Verify the source before finishing."}]}'
                else:
                    import re
                    text = messages[1]['content']
                    i = re.search(r'fixture (\d+)', text).group(1)
                    output = f'Action: Finish[VALUE_{i}]' if 'Reflections' in text else 'Action: Finish[wrong]'
                self.calls.append({'messages':copy.deepcopy(messages), 'output':output, 'purpose':purpose,
                                   'prompt_tokens':1,'output_tokens':1,'seconds':0.0})
                return output
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            data = Path(tmp)/'data.json'
            save(data,rows)
            cfg = {'out':str(Path(tmp)/'run'),'data':str(data),'environment':'local','seed':42,
                   'folds':2,'fold':0,'train_limit':2,'eval_limit':1,'demos':1,'retries':1,
                   'max_steps':3,'success_batch':2,'max_rules':10,'retriever':'bm25','top_k':1}
            exp = Experiment(cfg,FixtureModel())
            exp.collect(); exp.extract()
            state_path = Path(cfg['out'])/'insights.json'
            state = state_path.read_bytes()
            exp.evaluate()
            self.assertEqual(state, state_path.read_bytes())
            self.assertEqual(len(load(Path(cfg['out'])/'evaluation.json')['rows']),4)
            self.assertTrue((Path(cfg['out'])/'report.md').exists())
            self.assertEqual(load(state_path)['contrast_groups'],2)
            restarted = Experiment(cfg,FixtureModel())
            restarted.collect(); restarted.extract(); restarted.evaluate()
            self.assertEqual(restarted.model.calls, [])
            altered = dict(cfg,seed=10)
            with self.assertRaises(RuntimeError): Experiment(altered,FixtureModel())


if __name__ == '__main__':
    unittest.main()
