# %%
from src.probabilistic_model import ProbabilisticModel
from src.text_transform import load_text_transform
from tokenizer_and_save import text

import os

hash_to_texts = dict([(str(hash(t)), t) for t, _ in text])
texts_to_tokens = load_text_transform()
texts_to_tokens = dict(list(texts_to_tokens.items())[0:10])


# %%

model = ProbabilisticModel()

# %%
model.fit(texts_to_tokens)

# %%
model.save_model('dumps')

# %%
model.load_model('dumps')
# %%
