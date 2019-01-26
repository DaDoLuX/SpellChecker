from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import re
from collections import Counter


def _read_bigrams_and_probabilities(bigram_file):
    fp = open(bigram_file, 'r')
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for line in fp:
        line_text = line.strip().lower().split(',')
        text_list = line_text[0].split(' ')
        for w1, w2 in bigrams(text_list, pad_right=True, pad_left=True):
            value = int(line_text[-1:][0])
            model[w1][w2] = int(value)
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count
    fp.close()

    return model

#TODO ridurre conteggio
def _get_totale_words_value(unigram_file):
    fp = open(unigram_file, 'r')
    set_words = set()
    words = dict()
    for line in fp:
        line_text = line.split(',')
        set_words.add(line_text[0])

        key = line_text[0]
        value = int(line_text[-1:][0])
        words[key] = value
    fp.close()
    return set_words, words, sum(words.values())


model = _read_bigrams_and_probabilities('../Data/conteggio_bigram_catalogo_nltk.txt')
set_words, WORDS, N = _get_totale_words_value("../Data/conteggio_catalogo_nltk.txt")

def P(word, N=N):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    words = word.split()
    if len(words) == 1:
        # return max(candidates(word), key=P)
        return sorted(candidates(word), key=P, reverse=True)
    else:
        first_word = words[-2:-1][0]
        last_word = words[-1:][0]
        candidates_list = candidates(last_word)
        bigram_candidates = dict()
        for c in candidates_list:
            bigram_candidates[c] = model[first_word][c] * (WORDS[c] / N)

        return sorted(bigram_candidates.items(), key=lambda x:x[1],reverse=True)


def candidates(word):
    "Generate possible spelling corrections for word."
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


if __name__ == "__main__":

    print(correction("anel"))
    print(correction("acciaio zinca"))