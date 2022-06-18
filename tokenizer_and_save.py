# %%
from sklearn.model_selection import ShuffleSplit
from src.text_transform import text_transform
from submodule.data import cu, tw
import os

text = cu.get_text(f'{os.getcwd()}/submodule/data/') + \
    tw.get_text(f'{os.getcwd()}/submodule/data/')

# %%
split = ShuffleSplit(n_splits=1, test_size=.7)
text = [text[index] for index in next(split.split(text))[0]]
print(len(text))
# %%
if __name__ == '__main__':
    _ = text_transform([t for t, _ in text])

# %%

# %%
