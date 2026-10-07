# Attribution and licensing

This is an independent local adaptation of ExpeL, not the authors' official release.

Upstream: ExpeL: LLM Agents Are Experiential Learners, Andrew Zhao, Daniel Huang,
Quentin Xu, Matthieu Lin, Yong-Jin Liu and Gao Huang, AAAI 2024.
Repository: https://github.com/LeapLabTHU/ExpeL
Pinned revision: e41ec9a24823e7b560c561ab191441b56d9bcefc.

`data/paper_demonstrations.json` contains the six FEWSHOTS string literals extracted
from upstream `prompts/hotpotQA.py`. These examples originate from the ReAct line
of work and are retained with attribution. The upstream Apache-2.0 license is
included as `LICENSE-ExpeL.txt`. They were converted from Python literals to JSON;
wording was not newly fabricated. Embedded Wikipedia content retains its applicable license.

`data/hotpotqa_upstream_sample.json` is converted from the official ExpeL repository's
`data/hotpotqa/hotpot-qa-distractor-sample.joblib` (100 rows), preserving IDs, questions,
answers, contexts and labels. This conversion changes serialization only. Runtime
uses JSON and does not deserialize pickle files.

HotpotQA: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
Ruslan Salakhutdinov, Christopher D. Manning. EMNLP 2018.
Dataset homepage: https://hotpotqa.github.io/
Dataset and Wikipedia corpus license: **CC BY-SA 4.0**
https://creativecommons.org/licenses/by-sa/4.0/
The converted dataset is redistributed under that same license. This notice is not
a claim of authorship over the dataset or prompts.

New local implementation code follows the Apache-2.0 terms in LICENSE-ExpeL.txt.

## WebShop adaptation

`data/webshop_tasks.json` is transformed from ExpeL's `data/webshop/webshop.fixed100.json` at the pinned revision above. Only public instructions and session indices are retained; answer/product key fields are removed. `data/webshop_demonstrations.json` preserves the two FEWSHOTS strings from `prompts/webshop.py`, with upstream attribution and Apache-2.0 terms. Source hashes are recorded in `data/WEBSHOP_SOURCE.json`.

WebShop: Shunyu Yao, Howard Chen, John Yang, Karthik Narasimhan. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents, NeurIPS 2022. Official environment: https://github.com/princeton-nlp/WebShop (MIT). The external server and product catalogue are not bundled. The new HTTP adapter is an independent implementation; installing the external environment remains subject to its own license and data terms.
