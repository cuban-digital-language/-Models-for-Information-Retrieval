import codecs
from math import log
import json

import numpy as np

try:
    from .text_transform import get_progressbar
except (ModuleNotFoundError, ImportError):
    from text_transform import get_progressbar


class ProbabilisticModel:
    def __init__(self) -> None:
        self.corpus = []
        self.inverted_index = {}
        self.term_to_index = {}
        self.N = 0
        self.pi = []

    def __add_ii__(self, dj, term):
        try:
            self.inverted_index[term].add(dj)
        except KeyError:
            self.inverted_index[term] = set([dj])

    def computing_independent_values(self):
        print(" malloc document vectors ")
        self.document_w_vector = np.zeros((self.N, len(self.inverted_index)))
        # self.document_w_vector = [
        #     [0] * len(self.inverted_index) for _ in range(self.N)]
        bar = get_progressbar(self.N, ' precomputing weights ')
        bar.start()

        for key in self.inverted_index:
            l = len(self.inverted_index[key])
            for i in self.inverted_index[key]:
                pi, ri = self.pi[self.term_to_index[key]], log(
                    self.N/l, self.N + 1)

                value = log(
                    (pi*(1-ri) + 1)/(ri*(1-pi) + 1))
                self.document_w_vector[i][self.term_to_index[key]] = value

            bar.update(i + 1)
        bar.finish()

    def fit(self, texts):
        self.N = len(texts)
        bar = get_progressbar(len(texts), ' precomputing all values ')
        bar.start()
        for i, hsh in enumerate(texts):
            for token in texts[hsh]:
                self.__add_ii__(i, token.lower())
            bar.update(i + 1)
            self.corpus.append(hsh)
        bar.finish()

        bar = get_progressbar(len(self.inverted_index),
                              ' numerate terms ')
        bar.start()
        for i, key in enumerate(self.inverted_index):
            self.term_to_index[key] = i
            bar.update(i + 1)
        bar.finish()

        self.pi = [0.5] * len(self.inverted_index)

        self.computing_independent_values()

    def sorted_and_find(self, query, recover_len=10):
        query_term = set()
        for token in query:
            try:
                query_term.add(self.term_to_index[token.lower()])
            except:
                pass

        sim_result = []
        for i in range(self.N):
            sim_result.append(
                (i, sum([self.document_w_vector[i][j] for j in query_term])))
        sim_result.sort(key=lambda x: x[1], reverse=True)

        _len_ = recover_len if self.N > recover_len else self.N
        return [self.corpus[sim_result[i][0]] for i in range(_len_)]

    def save_model(self, path):
        # bar = get_progressbar(self.N, ' save doc vec ')
        # bar.start()
        # n = []
        # for i, v in enumerate(self.document_w_vector):
        #     n.append(tuple(v))
        #     bar.update(i + 1)
        # bar.finish()

        with open(f'{path}/data_from_vectorial_model.json', 'w+') as f:
            f.write(json.dumps({
                'ii': self.term_to_index,
                # 'd2v': tuple(n),
                'corpus': self.corpus,
                'N': self.N
            }))

            f.close()

        b = self.document_w_vector.tolist()  # nested lists with same data, indices
        file_path = f"{path}/probabilistic_doc_vec.json"  # your path variable
        json.dump(b, codecs.open(file_path, 'w+', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)  # this saves the array in .json format

    def load_model(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'r') as f:
            data = json.load(f)
            self.term_to_index = data['ii']
            # self.document_w_vector = data['d2v']
            self.corpus = data['corpus']
            self.N = data['N']

        file_path = f"{path}/probabilistic_doc_vec.json"  # your path variable
        obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        self.document_w_vector = np.array(b_new)

# a = np.arange(10).reshape(2, 5)  # a 2 by 5 array
# b = a.tolist()  # nested lists with same data, indices
# file_path = "/path.json"  # your path variable
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'),
#           separators=(',', ':'),
#           sort_keys=True,
#           indent=4)  # this saves the array in .json format
