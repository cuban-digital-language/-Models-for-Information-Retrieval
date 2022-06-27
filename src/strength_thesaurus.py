import json
from unittest import result
import numpy as np


try:
    from .text_transform import get_progressbar
except (ModuleNotFoundError, ImportError):
    from text_transform import get_progressbar


class StrengthThesaurus:
    def __init__(self, alpha=0.5, length=10) -> None:
        self.alpha = alpha
        self.length = length
        self._strength_ = {}

    def __getitem__(self, index: tuple) -> float:
        j, i = index
        return self._strength_[j][i]

    def __len__(self):
        return len(self.vocabulary)

    def expansion_query(self, query):
        terms = set()
        for term in query:
            words = self.expansion_term(term)
            terms = terms.union(words)

        return list(terms)

    def expansion_term(self, term):
        if not term in self.vocabulary:
            return []

        bar = get_progressbar(len(self._list_), f' expansion {term} ')
        bar.start()
        index = self._list_.index(term)
        result = []
        for j, _ in enumerate(self._list_):
            value = self.strength(index, j)
            result.append((j, value))
            bar.update(j+1)
        bar.finish()

        result.sort(key=lambda x: x[1], reverse=True)
        result = [self._list_[i] for i, value in result if value >= self.alpha]
        return result[0:self.length]

    def strength(self, tj, ti):
        try:
            return self._strength_[(tj, ti)]
        except KeyError:
            Dtj: set = set(self.vocabulary[self._list_[tj]]['X'])
            Dti: set = set(self.vocabulary[self._list_[ti]]['Y'])

            if Dti.isdisjoint(Dtj):
                return 0

            nij = Dti.intersection(Dtj)
            self._strength_[(tj, ti)] = (len(nij) + 1) / (len(Dti) + 2)
            return self._strength_[(tj, ti)]

    def fit(self, X, Y):
        self.vocabulary = {}

        for i, text in enumerate(X):
            for word in text:
                try:
                    self.vocabulary[word]['X'].append(i)
                except KeyError:
                    self.vocabulary[word] = {'X': [], 'Y': []}
                    self.vocabulary[word]['X'].append(i)

        for i, text in enumerate(Y):
            for word in text:
                try:
                    self.vocabulary[word]['Y'].append(i)
                except KeyError:
                    self.vocabulary[word] = {'X': [], 'Y': []}
                    self.vocabulary[word]['Y'].append(i)

        self._list_ = list(self.vocabulary)

    def dumps_path(self, path, key=""):
        return f'{path}/{key}/data_strength_t.json'

    def save_model(self, path, key=""):
        with open(self.dumps_path(path, key), 'w+') as f:
            f.write(json.dumps({
                'v': self.vocabulary,
                'l': self._list_,
            }))

    def load_model(self, path, key=''):
        with open(self.dumps_path(path, key), 'r') as f:
            data = json.load(f)
            self.vocabulary = data['v']
            self._list_ = data['l']


def print_expansion(query):
    for term in query:
        print(term, end=" ")
