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

        bar = get_progressbar(len(_len_), ' probability computer ')
        bar.start()
        for i, text in enumerate(self.strength_thesaurus._list_):
            tuple_.append((text, self.probability(i, distribution)))
            bar.update(i+1)
        bar.finish()

        tuple_ = [(x, y) for x, y in tuple_ if y > 1/_len_ and y > self.alpha]
        tuple_.sort(key=lambda x: x[1], reverse=True)
        return tuple_

    def probability(self, index, distribution):
        return self.beta * distribution[index] + ((1-self.beta) / (self.sj[index] + 1)) * sum([
            self.strength_thesaurus[(index, i)] * distribution[i]
            for _, i in self.edge_dict[index] if index != i
        ])

    def fit(self, X, Y):
        self.edge_dict = {}
        _len_ = len(self.strength_thesaurus)
        bar = get_progressbar(_len_ * _len_, ' term linked ')
        bar.start()
        for i in range(_len_):
            l = []
            for j in range(_len_):
                if i == j:
                    continue
                elif self.strength_thesaurus[(i, j)] > self.alpha:
                    l.append((self.strength_thesaurus[(i, j)], j))
                self.edge_dict[i] = l
                bar.update(i+1)
        bar.finish()

        self.sj = [
            sum([self.strength_thesaurus[(index, i)]
                for _, i in self.edge_dict[index] if index != i])
            for index in range(_len_)
        ]

    def save_model(self, path):
        with open(f'{path}/data_bayesian_t.json', 'w+') as f:
            f.write(json.dumps({
                'e': self.edge_dict,
                'sj': tuple(self.sj),
            }))

    def load_model(self, path):
        with open(f'{path}/data_bayesian_t.json', 'r') as f:
            data = json.load(f)
            self.edge_dict = data['e']
            self.sj = data['sj']

        self.strength_thesaurus.load_model(path)
