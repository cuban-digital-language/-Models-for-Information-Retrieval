# %%
from src.strength_thesaurus import StrengthThesaurus
from src.text_transform import load_text_transform
from submodule.data import get_all_text
from tokenizer_and_save import text
import os

hash_to_texts = dict([(str(hash(t)), t) for t, _ in text])
texts_to_tokens = load_text_transform()
# texts_to_tokens = dict(list(texts_to_tokens.items())[0:10])

# %%
model = StrengthThesaurus()

# %%
data = list(texts_to_tokens.values())
model.fit(data, data)

# %%
model.save_model('dumps')

# %%
