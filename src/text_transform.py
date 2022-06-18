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
                    token.text if token.lemma is None else token.lemma)

            for ent_named in nlp.__ents__(text):
                token_text.append(ent_named.text)

            corpus[str(hash(text))] = token_text
        bar.update(i+1)

    bar.finish()

    with open('dumps/token_text.json', 'w+') as f:
        f.write(json.dumps(corpus))

    return corpus


def load_text_transform():
    try:
        f = open('dumps/token_text.json', 'r')
        data = json.load(f)
    except:
        return False

    return data
