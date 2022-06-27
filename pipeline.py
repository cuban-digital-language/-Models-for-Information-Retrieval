# %%
import json

from sklearn import metrics
from src.bayesian_thesaurus import BayesianThesaurus
from src.probabilistic_model import ProbabilisticModel
from src.text_transform import pretty, text_transform
from src.strength_thesaurus import StrengthThesaurus, print_expansion
from submodule.data import cu, tw, fb, tg
import os
import sys
from src.vectorial_model import VectorialModel
# from tokenizer_and_save import text
import os
from sklearn.neighbors import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle


if len(sys.argv) != 2:
    raise "Expected social key (cu, tw, fb, tg)"

key = sys.argv[1]

# key = 'fb'
# print(f"Write your query to {key}")
# query = input()
query = 'Español Digital Cubano'
queryText = query
dataset = None

if key == 'cu':
    dataset = cu

if key == 'tw':
    dataset = tw

if key == 'fb':
    dataset = fb

if key == 'tg':
    dataset = tg


_text = dataset.get_text(f'{os.getcwd()}/submodule/data/')
print("Dataset size", len(_text))

# %%

try:
    f = open(f'dumps/{key}/token_text.json', 'x')
    f.close()
    corpus, real_text, vectors = text_transform(
        [t for t, _ in _text if any(t)])

    hshs = list(vectors.values())
    embedding = np.array(list(vectors.keys()))
    with open(f'dumps/{key}/token_text.json', 'w+') as f:
        f.write(json.dumps(corpus))
        f.close()
    with open(f'dumps/{key}/real_text.json', 'w+') as f:
        f.write(json.dumps(real_text))
        f.close()
    # with open(f'dumps/{key}/vectors_text.npy', 'w+') as f:
    #     f.close()
    #     np.save(f'dumps/{key}/vectors_text.npy', vectors)
    with open(f'dumps/{key}/hshs_text.json', 'w+') as f:
        f.write(json.dumps(hshs))
        f.close()

except OSError:
    f = open(f'dumps/{key}/token_text.json', 'r')
    corpus = json.load(f)
    f.close()
    f = open(f'dumps/{key}/real_text.json', 'r')
    real_text = json.load(f)
    f.close()
    # vectors = np.load(f'dumps/{key}/vectors_text.npy', allow_pickle=True)
    # f.close()
    f = open(f'dumps/{key}/hshs_text.json', 'r')
    hshs = json.load(f)
    f.close()


# %%

print("KDTree initialize")
try:
    f = open(f'dumps/{key}/classifier', 'x')
    f.close

    tfid = TfidfVectorizer()
    vectors = tfid.fit_transform([t for t, _ in _text if any(t)])
    svd = TruncatedSVD(96)
    vectors = svd.fit_transform(vectors)
    tree = KDTree(vectors)
    embedding = KDTree(embedding)
    with open(f'dumps/{key}/classifier', 'wb') as f:
        f.write(pickle.dumps(tree))
        f.close()
    with open(f'dumps/{key}/embedding', 'wb') as f:
        f.write(pickle.dumps(embedding))
        f.close()
    with open(f'dumps/{key}/vectorized', 'wb') as f:
        f.write(pickle.dumps(tfid))
        f.close()
    with open(f'dumps/{key}/svd', 'wb') as f:
        f.write(pickle.dumps(svd))
        f.close()

except OSError:
    f = open(f'dumps/{key}/classifier', 'rb')
    tree = pickle.loads(f.read())
    f.close
    f = open(f'dumps/{key}/embedding', 'rb')
    embedding = pickle.loads(f.read())
    f.close
    f = open(f'dumps/{key}/vectorized', 'rb')
    tfid = pickle.loads(f.read())
    f.close
    f = open(f'dumps/{key}/svd', 'rb')
    svd = pickle.loads(f.read())
    f.close

# %%

q, _, qv = text_transform([query])
query_embedding = list(qv.keys())
query_vector = tfid.transform([query])
query_vector = svd.transform(query_vector)
query = q[str(hash(query))]

# %%
d, ind = tree.query(query_vector, k=10)
print(ind[0])
tree_result = [_text[i] for i in ind[0]]

print()
for i, hsh in enumerate(tree_result):
    text = pretty(query, hsh[0])[0]
    tree_result[i] = hsh[0]

    print()
    print(f'Ranking #: {i + 1} d: {d[0][i]}')
    print("\n#####################################################################\n")
    print(text)

# %%
d, ind = embedding.query(query_embedding, k=10)
embedding_result = [hshs[i] for i in ind[0]]

print()
for i, hsh in enumerate(embedding_result):
    text = pretty(query, real_text[hsh])[0]
    embedding_result[i] = real_text[hsh]

    print()
    print(f'Ranking #: {i + 1} d: {d[0][i]}')
    print("\n#####################################################################\n")
    print(text)


# %%

v_model = VectorialModel()

try:
    f = open(v_model.dumps_path('dumps', key), 'x')
    f.close()
    v_model.fit(corpus)
    v_model.save_model('dumps', key)
except OSError:
    v_model.load_model('dumps', key)


# %%
vectorial_result = v_model.get_ranking(query)

print()
for i, hsh in enumerate(vectorial_result):
    text = pretty(query, real_text[hsh])[0]
    vectorial_result[i] = real_text[hsh]

    print()
    print(f'Ranking #: {i + 1}')
    print("\n#####################################################################\n")
    print(text)


# %%

p_model = ProbabilisticModel()

try:
    f = open(p_model.dumps_path('dumps', key), 'x')
    f.close()
    p_model.fit(corpus)
    p_model.save_model('dumps', key)
except OSError:
    p_model.load_model('dumps', key)
    p_model.computing_independent_values()

# %%
pb_result = p_model.sorted_and_find(query)

for i, hsh in enumerate(pb_result):
    text = pretty(query, real_text[hsh])[0]
    pb_result[i] = real_text[hsh]

    print()
    print(f'Ranking #: {i + 1}')
    print("\n#####################################################################\n")
    print(text)

# %%

s_thesaurus = StrengthThesaurus(alpha=0.8)


try:
    f = open(s_thesaurus.dumps_path('dumps', key), 'x')
    f.close()
    data = list(corpus.values())
    s_thesaurus.fit(data, data)
    s_thesaurus.save_model('dumps', key)
except OSError:
    s_thesaurus.load_model('dumps', key)

# %%

ep_query = s_thesaurus.expansion_query(query)
print()
print_expansion(ep_query)

# %%
exp_vectorial_result = v_model.get_ranking(ep_query)

for i, hsh in enumerate(exp_vectorial_result):
    text = pretty(ep_query, real_text[hsh])[0]
    exp_vectorial_result[i] = real_text[hsh]

    print()
    print(f'Ranking #: {i + 1}')
    print("\n#####################################################################\n")
    print(text)


# %%
b_thesaurus = BayesianThesaurus(alpha=0.2)
b_thesaurus.strength_thesaurus = s_thesaurus

try:
    # f = open(b_thesaurus.dumps_path('dumps', key), 'x')
    # f.close()
    data = list(corpus.values())
    b_thesaurus.fit(data, data)
    b_thesaurus.save_model('dumps', key)
except OSError:
    b_thesaurus.load_model('dumps', key)

# %%

ep_query = b_thesaurus.expansion(query)
print()
print_expansion(ep_query)
ep_query = [w for w, _ in ep_query]

# %%
exp_pp_result = p_model.sorted_and_find(ep_query)

for i, hsh in enumerate(exp_pp_result):
    text = pretty(ep_query, real_text[hsh])[0]
    exp_pp_result[i] = real_text[hsh]
    print()
    print(f'Ranking #: {i + 1}')
    print("\n#####################################################################\n")
    print(text)


# %%

with open(f"results/{queryText}-{key}.json", 'w+') as f:
    f.write(json.dumps({
        'v': vectorial_result,
        'p': pb_result,
        'sv': exp_vectorial_result,
        'tt': tree_result
    }))

    f.close()
