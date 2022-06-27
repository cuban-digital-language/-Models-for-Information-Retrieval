import json
import progressbar

try:
    from ..submodule.tokenizer.custom_tokenizer import SpacyCustomTokenizer
except (ModuleNotFoundError, ImportError):
    from submodule.tokenizer.custom_tokenizer import SpacyCustomTokenizer


def get_progressbar(N, name=""):
    return progressbar.ProgressBar(
        maxval=N,
        widgets=[progressbar.Bar('#', '[', ']'),
                 name,
                 progressbar.Percentage()])


def text_transform(texts):
    corpus = {}
    real_text = {}
    vector_text = {}
    nlp = SpacyCustomTokenizer()
    bar = get_progressbar(len(texts), f' {len(texts)} corpus tokenizer ')
    bar.start()
    for i, text in enumerate(texts):

        if not hash(text) in corpus:
            token_text = []
            for token in nlp(text):
                if token.is_stop or token.is_symbol or token.is_url() or token.is_emoji() or not any(token.text) or token.space():
                    continue

                token_text.append(
                    token.text.lower() if token.lemma is None else token.lemma.lower())

            for ent_named in nlp.__ents__(text):
                token_text.append(ent_named.text.lower())

            real_text[str(hash(text))] = text
            corpus[str(hash(text))] = token_text
            vector_text[tuple(nlp.nlp(text).vector)] = str(hash(text))
        bar.update(i+1)

    bar.finish()

    return corpus, real_text, vector_text


def load_text_transform():
    f = open('dumps/token_text.json', 'r')
    data = json.load(f)
    f.close()
    f = open('dumps/real_text.json', 'r')
    data2 = json.load(f)
    f.close()
    return data, data2


def pretty(q, *texts):
    result = list(texts)
    for key in q:
        for i, text in enumerate(result):
            lower_text: str = text.lower()
            index = 0
            while True:
                try:
                    index = lower_text.index(key, index)
                    result[i] = text[0:index] + "\33[46m{}\033[00m".format(
                        text[index: index + len(key)]) + text[index + len(key) + 1:]
                    index += len(key)
                except ValueError:
                    break
    return tuple(result)
