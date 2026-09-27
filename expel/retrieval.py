import math
import re
from collections import Counter


def tokens(text):
    stopwords = {'a','an','the','is','are','was','were','in','on','of','to','and','or','for',
                 'with','what','which','who','where','when','how','does','do','did'}
    return [t for t in re.findall(r'\w+', text.lower()) if t not in stopwords]


class BM25:
    def __init__(self, documents):
        self.docs = [Counter(tokens(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = sum(self.lengths) / max(1, len(self.docs)) or 1
        self.df = Counter(t for d in self.docs for t in d)

    def rank(self, query):
        ranked = []
        for index, doc in enumerate(self.docs):
            score = 0.0
            for token in set(tokens(query)):
                tf = doc[token]
                idf = math.log(1 + (len(self.docs) - self.df[token] + .5) / (self.df[token] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths[index] / self.average))
            ranked.append((score, index))
        return sorted(ranked, key=lambda pair: (-pair[0], pair[1]))


class ExperienceRetriever:
    def __init__(self, episodes, method='bm25'):
        self.episodes = episodes
        self.method = method
        questions = [e['question'] for e in episodes]
        if method == 'mpnet':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
            self.vectors = self.encoder.encode(questions, normalize_embeddings=True) if questions else None
        else:
            self.index = BM25(questions)

    def retrieve(self, question, k):
        if not self.episodes:
            return []
        if self.method == 'mpnet':
            vector = self.encoder.encode([question], normalize_embeddings=True)[0]
            ranked = sorted(enumerate(self.vectors @ vector), key=lambda p: -float(p[1]))
            return [self.episodes[i] for i, _ in ranked[:k]]
        return [self.episodes[i] for _, i in self.index.rank(question)[:k]]
