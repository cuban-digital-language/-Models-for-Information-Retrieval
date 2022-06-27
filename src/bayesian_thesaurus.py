import json


try:
    from .text_transform import get_progressbar
    from .strength_thesaurus import StrengthThesaurus
except (ModuleNotFoundError, ImportError):
    from text_transform import get_progressbar
    from strength_thesaurus import StrengthThesaurus


class BayesianThesaurus:
    def __init__(self, alpha=0.5, beta=0.8) -> None:
        self.strength_thesaurus = StrengthThesaurus()
        self.alpha = alpha
        self.beta = beta

    def expansion(self, query):
        _len_ = len(self.strength_thesaurus)
        distribution = []
        for word in self.strength_thesaurus._list_:
            if word in query:
                distribution.append(1)
            else:
                distribution.append(1/_len_)

        bar = get_progressbar(_len_, ' probability computer ')
        bar.start()
        tuple_ = []
        for i, text in enumerate(self.strength_thesaurus._list_):
            tuple_.append((text, self.probability(i, distribution)))
            bar.update(i+1)
        bar.finish()

        tuple_.sort(key=lambda x: x[1], reverse=True)
        tuple_ = [(x, y) for x, y in tuple_ if y > 1/_len_ and y > self.alpha]
        return tuple_

    def probability(self, index, distribution):
        return self.beta * distribution[index] + ((1-self.beta) / (self.sj[index] + 1)) * sum([
            self.strength_thesaurus.strength(index, i) * distribution[i]
            for _, i in self.edge_dict[str(index)] if index != i
        ])

    def fit(self, X, Y):
        self.edge_dict = {}
        _len_ = len(self.strength_thesaurus)
        bar = get_progressbar(_len_ * _len_, f' {_len_ * _len_} term linked ')
        bar.start()
        index = 0
        for i in range(_len_):
            l = []
            for j in range(_len_):
                if i == j:
                    continue
                elif self.strength_thesaurus.strength(i, j) > self.alpha:
                    l.append((self.strength_thesaurus.strength(i, j), j))
                index += 1
                bar.update(index)
            self.edge_dict[str(i)] = l
        bar.finish()

        self.sj = [
            sum([self.strength_thesaurus.strength(i, j)
                for _, i in self.edge_dict[str(index)] if index != i])
            for index in range(_len_)
        ]

    def dumps_path(self, path, key=""):
        return f'{path}/{key}/data_bayesian_t.json'

    def save_model(self, path, key=""):
        with open(self.dumps_path(path, key), 'w+') as f:
            f.write(json.dumps({
                'e': self.edge_dict,
                'sj': tuple(self.sj),
            }))

    def load_model(self, path, key=""):
        with open(self.dumps_path(path, key), 'r') as f:
            data = json.load(f)
            self.edge_dict = data['e']
            self.sj = data['sj']
