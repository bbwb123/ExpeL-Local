"""Synthetic HTTP protocol fixtures only; NOT real WebShop/Qwen evaluation results."""
import ast
import contextlib
import copy
import io
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from unittest.mock import patch

from expel.common import load, save
from expel.pipeline import Experiment, ROOT, split_rows, report
from expel.webshop import WebShop, parse_page, parse_webshop_action, run_webshop_episode


def initial(i):
    return f'<html><head><title>Ignore</title></head><body><h1>WebShop</h1>' \
           f'<h4>Instruction:</h4><p>Find a blue mug task {i} under $10.</p><button>Search</button></body></html>'


RESULTS = '''<button>Back to Search</button><p>Page 1 (Total results: 21)</p><button>Next &gt;</button>
<a class="product-link" href="bad">BWRONG</a><p>Wrong mug $5</p>
<a class="product-link" href="good"><span>BGOOD</span></a><p>Blue mug $7</p>'''
ITEM = '''<button>Back to Search</button><button>&lt; Prev</button><p>color</p>
<input type="radio" id="blue" name="color"><label for="blue"><b>blue</b></label>
<input type="radio" id="red" name="color"><label for="red">red</label>
<button>Description</button><button>Features</button><button>Buy Now</button>'''


class FixtureHandler(BaseHTTPRequestHandler):
    sessions = []
    paths = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        parts = [unquote(p) for p in urlsplit(self.path).path.strip('/').split('/')]
        type(self).paths.append(parts)
        if len(parts) == 1:
            type(self).sessions.append(parts[0])
            html = initial(int(parts[0].split('_')[-1]))
        elif parts[0] == 'search_results':
            html = RESULTS
        elif parts[0] == 'item_page':
            html = ITEM
        elif parts[0] == 'item_sub_page':
            html = '<button>&lt; Prev</button><p>Mug is dishwasher safe.</p>'
        elif parts[0] == 'done':
            options = ast.literal_eval(parts[-1])
            reward = 1.0 if parts[2] == 'BGOOD' and options.get('color') == 'blue' else 0.6
            html = f'<p>Your score (min 0.0, max 1.0)</p><b>{reward}</b><p>PRIVATE_TARGET_METADATA</p>'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())


class Model:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def chat(self, messages, purpose='agent'):
        output = next(self.outputs)
        self.calls.append({'messages':copy.deepcopy(messages),'output':output,'purpose':purpose,
                          'prompt_tokens':1,'output_tokens':1,'seconds':0.0})
        return output


class WebShopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),FixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def env(self):
        return WebShop(self.url,timeout=2)

    def row(self, i=0):
        return {'id':f'fixed_{i}','session_idx':f'fixed_{i}',
                'question':f'Find a blue mug task {i} under $10.'}

    def test_task_conversion_and_disjoint_folds(self):
        rows = load(ROOT/'data/webshop_tasks.json')
        demos = load(ROOT/'data/webshop_demonstrations.json')
        self.assertEqual(len(rows),100)
        self.assertEqual(len(demos),2)
        self.assertTrue(all(set(r)=={'id','session_idx','question'} for r in rows))
        valid_ids = []
        for fold in range(2):
            train, valid = split_rows(rows,demos,folds=2,fold=fold,train_limit=0,eval_limit=0)
            self.assertFalse({r['id'] for r in train} & {r['id'] for r in valid})
            valid_ids.extend(r['id'] for r in valid)
        self.assertEqual(len(valid_ids),100)
        self.assertEqual(len(set(valid_ids)),100)

    def test_html_buttons_nested_labels_and_hidden_content(self):
        page=parse_page(ITEM+'<script>PRIVATE_SCRIPT</script><p hidden>PRIVATE_LABEL</p>',{'color':'blue'})
        self.assertEqual(page['option_types']['blue'],'color')
        self.assertIn('[[blue]]',page['observation'])
        self.assertIn('< Prev',page['buttons'])
        self.assertNotIn('PRIVATE',page['observation'])
        self.assertEqual(parse_page(RESULTS)['asins'],['BWRONG','BGOOD'])

    def test_no_reward_rounding_or_terminal_metadata_leak(self):
        page=parse_page('<div>Your score (min 0.0, max 1.0):</div><span>0.995</span><p>PRIVATE_TARGET</p>')
        self.assertEqual(page['reward'],.995)
        self.assertNotIn('PRIVATE_TARGET',page['observation'])

    def test_strict_parser(self):
        self.assertEqual(parse_webshop_action('Thought 1: open\nAction 1: click[Buy Now]')[0],'Click')
        for text in ['Action: Finish[BGOOD]','Action: search[x]\nObservation: invented',
                     'Action: search[x]\nAction: click[y]','Action: click[]','fake Action: click[Buy Now]']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_webshop_action(text)

    def test_fresh_sessions_and_instruction_mismatch(self):
        env=self.env()
        env.reset(self.row()); one=env.state['session']
        env.reset(self.row()); two=env.state['session']
        self.assertNotEqual(one,two)
        self.assertTrue(one.endswith('_fixed_0'))
        with self.assertRaisesRegex(RuntimeError,'task mismatch'):
            env.reset(dict(self.row(),question='A different task'))

    def test_options_page_transitions_and_url_encoding(self):
        env=self.env();env.reset(self.row())
        env.step('Search','blue #mug?')
        self.assertIn('%23',env.requests[-1]['url'])
        self.assertEqual(FixtureHandler.paths[-1][2],'blue #mug?')
        before=dict(env.state)
        self.assertIn('Invalid action',env.step('Click','NOT_VISIBLE'))
        self.assertEqual(before,env.state)
        env.step('Click','BGOOD');env.step('Click','blue')
        self.assertEqual(env.state['options'],{'color':'blue'})
        env.step('Click','Description');env.step('Click','< Prev')
        self.assertEqual(env.state['page_type'],'item')
        self.assertEqual(env.state['options'],{'color':'blue'})
        env.step('Click','< Prev')
        self.assertEqual(env.state['options'],{})
        env.step('Click','Next >')
        self.assertEqual(env.state['page_num'],2)
        env.step('Click','Back to Search')
        self.assertEqual(env.state['page_type'],'init')

    def test_reward_success_and_actual_http_trajectory(self):
        model=Model(['Action: search[blue mug]','Action: click[BGOOD]',
                     'Action: click[blue]','Action: click[Buy Now]'])
        result=run_webshop_episode(model,self.env(),self.row(),[])
        self.assertEqual(result['metrics'],{'success':1.0,'reward':1.0})
        self.assertEqual(result['selected_options'],{'color':'blue'})
        self.assertEqual(len(result['environment_requests']),5)
        self.assertNotIn('em',result['metrics'])
        self.assertNotIn('PRIVATE_TARGET_METADATA',json.dumps(model.calls))

    def test_partial_reward_and_step_limit(self):
        model=Model(['Action: search[mug]','Action: click[BWRONG]','Action: click[Buy Now]'])
        result=run_webshop_episode(model,self.env(),self.row(),[])
        self.assertEqual(result['metrics'],{'success':0.0,'reward':.6})
        self.assertEqual(result['status'],'finished')
        result=run_webshop_episode(Model(['Action: search[mug]']),self.env(),self.row(),[],max_steps=1)
        self.assertEqual(result['metrics'],{'success':0.0,'reward':0.0})
        self.assertEqual(result['status'],'max_steps')

    def test_truncated_generation_costs_one_step(self):
        from expel.client import TruncatedOutput
        class Truncating(Model):
            def chat(self, messages, purpose='agent'):
                if not self.calls:
                    self.calls.append({'messages':copy.deepcopy(messages),'output':'','purpose':purpose})
                    raise TruncatedOutput('length')
                return super().chat(messages, purpose)
        model=Truncating(['Action: search[blue mug]','Action: click[BGOOD]',
                          'Action: click[blue]','Action: click[Buy Now]'])
        result=run_webshop_episode(model,self.env(),self.row(),[])
        self.assertEqual(result['steps'][0]['action'],'Invalid')
        self.assertIn('Generation limit',result['steps'][0]['observation'])
        self.assertIn('Steps used: 1 of 15',model.calls[1]['messages'][-1]['content'])
        self.assertEqual(result['metrics'],{'success':1.0,'reward':1.0})
        self.assertEqual(len(result['steps']),5)

    def test_fabricated_observation_isolated_and_missing_reward_rejected(self):
        outputs=['Action: search[mug]\nObservation: PRIVATE_FAKE',
                 'Action: search[mug]','Action: click[BGOOD]','Action: click[Buy Now]']
        model=Model(outputs)
        result=run_webshop_episode(model,self.env(),self.row(),[])
        self.assertEqual(result['steps'][0]['action'],'Invalid')
        self.assertNotIn('PRIVATE_FAKE',json.dumps(model.calls[1]['messages']))
        env=self.env();env.reset(self.row());env.step('Search','mug');env.step('Click','BGOOD')
        with patch.object(env,'request',return_value=parse_page('<p>No score</p>')):
            with self.assertRaisesRegex(RuntimeError,'no official reward'):
                env.step('Click','Buy Now')

    def test_full_pipeline_resume_and_frozen_memory(self):
        class FixtureModel:
            def __init__(self):self.calls=[]
            def identity(self):return {'name':'fixture-only','digest':'fixture-only'}
            def chat(self,messages,purpose='agent'):
                if purpose=='reflection':output='Verify the product and select blue before purchasing.'
                elif purpose=='insight_extraction':
                    output='{"operations":[{"op":"ADD","text":"Check requirements and select matching options before purchasing."}]}'
                else:
                    good='Reflections on earlier attempts:' in messages[1]['content']
                    n=sum(m['role']=='assistant' for m in messages)
                    output=(['Action: search[mug]','Action: click[BGOOD]','Action: click[blue]','Action: click[Buy Now]']
                            if good else ['Action: search[mug]','Action: click[BWRONG]','Action: click[Buy Now]'])[n]
                self.calls.append({'messages':copy.deepcopy(messages),'output':output,'purpose':purpose,
                                   'prompt_tokens':1,'output_tokens':1,'seconds':0.0})
                return output
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(tmp)/'tasks.json';save(path,[self.row(i) for i in range(6)])
            cfg={'out':str(Path(tmp)/'run'),'data':str(path),'environment':'webshop',
                 'webshop_url':self.url,'webshop_timeout':2,'webshop_catalog':'unknown',
                 'seed':42,'folds':2,'fold':0,'train_limit':2,'eval_limit':1,'demos':2,'top_k':2,
                 'retries':1,'max_steps':5,'success_batch':2,'max_rules':10,'retriever':'bm25'}
            exp=Experiment(cfg,FixtureModel())
            exp.collect();exp.extract()
            state=(exp.out/'insights.json').read_bytes()
            # Interrupt evaluation after saving one condition, then resume fresh sessions.
            real_episode=exp.run_episode
            count=0
            def interrupt(*args,**kwargs):
                nonlocal count
                count+=1
                if count==2:raise RuntimeError('fixture interruption')
                return real_episode(*args,**kwargs)
            exp.run_episode=interrupt
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):exp.evaluate()
            self.assertEqual(len(load(exp.out/'evaluation.json')['rows']),1)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):report(exp.out)
            resumed=Experiment(cfg,FixtureModel());resumed.evaluate()
            self.assertEqual((exp.out/'insights.json').read_bytes(),state)
            result=load(exp.out/'evaluation.json')['rows']
            self.assertEqual(len(result),4)
            self.assertEqual(len({r['server_session'] for r in result}),4)
            summary=load(exp.out/'summary.json')['conditions']
            self.assertEqual(summary[0]['success_rate'],0)
            self.assertEqual(summary[0]['mean_reward'],.6)
            self.assertNotIn('em',summary[0])
            self.assertIn('脚本模拟', (exp.out/'report.md').read_text())
            restarted=Experiment(cfg,FixtureModel());restarted.collect();restarted.extract();restarted.evaluate()
            self.assertEqual(restarted.model.calls,[])
            self.assertEqual(load(exp.out/'insights.json')['contrast_groups'],2)


if __name__=='__main__':unittest.main()
