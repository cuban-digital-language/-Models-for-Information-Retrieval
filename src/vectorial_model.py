from math import log
import json

try:
    from .text_transform import get_progressbar
except (ModuleNotFoundError, ImportError):
    from text_transform import get_progressbar


class VectorialModel:
    def __init__(self, alpha=0.5, recover_len=10) -> None:
        self.alpha = alpha
        self.recover_len = recover_len

    def fit(self, corpus: dict):
        self.N = len(corpus)
        self.invected_index = {}
        self.max_freq_doc = {}

        bar = get_progressbar(self.N, ' term indexation ')
        bar.start()
        for i, text_hsh in enumerate(corpus):
            f_max = 0
            for token in corpus[text_hsh]:
                try:
                    doc_dic = self.invected_index[token]
                except KeyError:
                    doc_dic = {}

                try:
                    doc_dic[text_hsh] += 1
                except KeyError:
                    doc_dic[text_hsh] = 1

                if doc_dic[text_hsh] > f_max:
                    f_max = doc_dic[text_hsh]

                self.invected_index[token] = doc_dic

            self.max_freq_doc[text_hsh] = f_max

            bar.update(i + 1)
        bar.finish()

        self.doc2vec = {}
        self.doc_w2 = {}
        self.term_to_index = [key for key in self.invected_index]
        self.term_to_index.sort()

        bar = get_progressbar(
            len(self.term_to_index), f' document vectorization {len(self.term_to_index)} ')
        bar.start()
        for text_hsh in corpus:
            self.doc2vec[text_hsh] = {}

        for i, term in enumerate(self.term_to_index):
            for text_hsh in self.invected_index[term]:
                tf = self.invected_index[term][text_hsh] / \
                    self.max_freq_doc[text_hsh]
                idf = log(self.N/len(self.invected_index[term]))
                self.doc2vec[text_hsh][term] = (tf * idf)

            bar.update(i+1)
        bar.finish()

        bar = get_progressbar(
            len(corpus), f' document quadratic computing {len(corpus)} ')
        bar.start()
        for i, text_hsh in enumerate(corpus):
            self.doc_w2[text_hsh] = sum(
                [w*w for _, w in self.doc2vec[text_hsh].items()])
            bar.update(i+1)
        bar.finish()

    def get_ranking(self, query_tokens):
        fqi, max_fqi = {}, 0
        for word in query_tokens:
            if not word in self.invected_index:
                continue

            try:
                fqi[word] += 1
            except KeyError:
                fqi[word] = 1

            if max_fqi < fqi[word]:
                max_fqi = fqi[word]

        query_vector = []
        for word in fqi:
            w = (self.alpha + (1 - self.alpha) *
                 fqi[word]/max_fqi) * log(self.N/len(self.invected_index[word]))

            query_vector.append((word, w))

        q2 = sum([w*w for _, w in query_vector])

        ranking = []
        for text_hsh, term_w in self.doc2vec.items():
            tqw = 0
            for term, qw in query_vector:
                try:
                    tqw += term_w[term] * qw
                except KeyError:
                    continue

            ranking.append((tqw/(q2 * self.doc_w2[text_hsh]), text_hsh))

        ranking.sort(key=lambda x: x[0], reverse=True)
        return [ranking[i][1] for i in range(self.recover_len)]

    def save_model(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'w+') as f:
            f.write(json.dumps({
                'ii': self.invected_index,
                'd2v': self.doc2vec,
                'dw2': self.doc_w2,
                'N': self.N
            }))

            f.close()

    def load_model(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'r') as f:
            data = json.load(f)
            self.invected_index = data['ii']
            self.doc2vec = data['d2v']
            self.doc_w2 = data['dw2']
            self.N = data['N']
