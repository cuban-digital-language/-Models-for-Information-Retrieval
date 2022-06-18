import json
import numpy as np


try:
    from .text_transform import get_progressbar
except (ModuleNotFoundError, ImportError):
    from text_transform import get_progressbar


class StrengthThesaurus:
    def __init__(self, alpha=0.5) -> None:
        self.alpha = 0.5

    def __getitem__(self, index: tuple) -> float:
        j, i = index
        return self._strength_[j][i]

    def __len__(self):
        return len(self.vocabulary)

    def expansion(self, term):
        if not term in self.vocabulary:
            return []
        index = self._list_.index(term)
        return [self._list_[i] for i, value in enumerate(self._strength_[index]) if value >= self.alpha]

    def strength(self, tj, ti):
        Dtj: set = self.vocabulary[tj]['X']
        Dti: set = self.vocabulary[ti]['Y']

        if Dti.isdisjoint(Dtj):
            return 0

        nij = Dti.intersection(Dtj)
        return (len(nij) + 1) / (len(Dti) + 2)

    def fit(self, X, Y):
        self.vocabulary = {}

        for i, text in enumerate(X):
            for word in text:
                try:
                    self.vocabulary[word]['X'].add(i)
                except KeyError:
                    self.vocabulary[word] = {'X': set(), 'Y': set()}
                    self.vocabulary[word]['X'].add(i)

        for i, text in enumerate(Y):
            for word in text:
                try:
                    self.vocabulary[word]['Y'].add(i)
                except KeyError:
                    self.vocabulary[word] = {'X': set(), 'Y': set()}
                    self.vocabulary[word]['Y'].add(i)

        self._list_ = list(self.vocabulary)

        l = len(self._list_)
        self._strength_ = np.zeros((l, l))

        bar = get_progressbar(l * l, f' strength computer {l*l} ')
        bar.start()
        for j, text_j in enumerate(self._list_):
            for i, text_i in enumerate(self._list_):
                self._strength_[j][i] = self.strength(text_j, text_i)
            bar.update(j+1)
        bar.finish()

    def save_model(self, path):
        with open(f'{path}/data_strength_t.json', 'w+') as f:
            f.write(json.dumps({
                'v': tuple(self.vocabulary.keys()),
                'l': self._list_,
            }))
        np.save(f"{path}/strength_matrix.npy", self._strength_)

    def load_model(self, path):
        with open(f'{path}/data_strength_t.json', 'r') as f:
            data = json.load(f)
            self.vocabulary = data['v']
            self._list_ = data['l']
        self._strength_ = np.load(f"{path}/strength_matrix.npy")
