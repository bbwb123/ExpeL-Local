import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .common import fingerprint, load, save
from .retrieval import BM25


def corpus_from_rows(rows):
    # Only public paragraphs enter tools. Answers and supporting-fact labels do not.
    pages = {}
    for row in rows:
        context = row['context']
        pairs = zip(context['title'], context['sentences']) if isinstance(context, dict) else context
        for title, sentences in pairs:
            bucket = pages.setdefault(title, [])
            for sentence in sentences:
                if sentence.strip() and sentence.strip() not in bucket:
                    bucket.append(sentence.strip())
    return pages


class LocalWiki:
    """Fixed corpus of ALL sampled tasks' distractor paragraphs, not live Wikipedia."""
    def __init__(self, pages):
        self.pages, self.titles = pages, sorted(pages)
        self.title_map = {t.casefold(): t for t in self.titles}
        self.index = BM25([t + ' ' + ' '.join(pages[t]) for t in self.titles])
        self.reset()

    def reset(self):
        self.active = []
        self.title = ''
        self.cursors = {}

    def search(self, query):
        exact = self.title_map.get(query.strip().casefold())
        self.reset()
        if exact:
            self.title, self.active = exact, self.pages[exact]
            return exact + ': ' + ' '.join(self.active[:3])[:1800]
        results = [(s, i) for s, i in self.index.rank(query)[:5] if s > 0]
        if not results:
            return 'No matching page in the fixed corpus. Reformulate the query.'
        return 'No exact title. Search one of these titles:\n' + '\n'.join(
            self.titles[i] + ': ' + ' '.join(self.pages[self.titles[i]])[:220] for _, i in results)

    def lookup(self, keyword):
        if not self.active:
            return 'No active page. Search an exact title first.'
        key = keyword.casefold()
        matches = [s for s in self.active if key in s.casefold()]
        cursor = self.cursors.get(key, 0)
        if cursor >= len(matches):
            return f'No more matches for {keyword!r} in {self.title}.'
        self.cursors[key] = cursor + 1
        return f'({cursor+1}/{len(matches)}) {matches[cursor][:1800]}'


class LiveWiki(LocalWiki):
    """Optional live MediaWiki tools; cache is shared by all evaluation conditions."""
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.reset()

    def api(self, params):
        path = self.cache_dir / (fingerprint(params) + '.json')
        if path.exists():
            return load(path)
        query = urllib.parse.urlencode(dict(params, action='query', format='json', formatversion=2))
        request = urllib.request.Request('https://en.wikipedia.org/w/api.php?' + query,
                    headers={'User-Agent': 'ExpeL-Local/0.1 (research reproduction; Python urllib)'})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError('MediaWiki API: ' + str(result['error']))
        save(path, result)
        return result

    def search(self, query):
        self.reset()
        data = self.api({'titles': query, 'prop': 'extracts', 'explaintext': 1, 'redirects': 1})
        page = data.get('query', {}).get('pages', [{}])[0]
        if page.get('missing') or not page.get('extract'):
            data = self.api({'list': 'search', 'srsearch': query, 'srlimit': 5})
            return 'Search one of these exact titles: ' + ', '.join(p['title'] for p in data.get('query', {}).get('search', []))
        self.title = page['title']
        self.active = [s.strip() for s in re.split(r'(?<=[.!?])\s+', page['extract']) if s.strip()]
        return self.title + ': ' + ' '.join(self.active[:3])[:1800]
