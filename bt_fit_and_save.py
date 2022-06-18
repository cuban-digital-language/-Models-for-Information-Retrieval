# %%
from src.bayesian_thesaurus import BayesianThesaurus
from src.text_transform import load_text_transform
from tokenizer_and_save import text

hash_to_texts = dict([(str(hash(t)), t) for t, _ in text])
texts_to_tokens = load_text_transform()
# texts_to_tokens = dict(list(texts_to_tokens.items())[0:10])

# %%
model = BayesianThesaurus()

# %%
data = list(texts_to_tokens.values())
model.strength_thesaurus.load_model('dumps')
model.fit(data, data)

# %%
model.save_model('dumps')

# %%
