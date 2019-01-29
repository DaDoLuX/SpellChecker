from nltk import bigrams, trigrams
from collections import defaultdict
import copy


NGRAMS_CATALOGUE = '../Data/conteggio_bigram_catalogo_nltk.txt'
UNIGRAM_CATALOGUE = '../Data/conteggio_catalogo_nltk.txt'


def _read_bigrams_and_probabilities(bigram_file):
    file = open(bigram_file, 'r')
    product_model = defaultdict(lambda: defaultdict(lambda: 0))
    total = 0
    for line in file:
        line_text = line.strip().lower().split(',')
        words_list = line_text[0].split(' ')
        value = int(line_text[-1:][0])
        total += value
        for w1, w2 in bigrams(words_list, pad_right=False, pad_left=True):
            product_model[w1][w2] = int(value)
    generic_model = copy.deepcopy(product_model)

    for w1 in product_model:
        total_count = float(sum(product_model[w1].values()))
        for w2 in product_model[w1]:
            product_model[w1][w2] /= total_count
    for w1 in generic_model:
        for w2 in generic_model[w1]:
            generic_model[w1][w2] /= total
    file.close()

    return product_model, generic_model


product_model, generic_model = \
    _read_bigrams_and_probabilities(NGRAMS_CATALOGUE)

def _generate_catalogue_dictionary(file):
    set_words = set()
    file = open(file, 'r')
    for line in file:
        line_text = line.strip().lower().split(',')
        words_list = line_text[0].split(' ')
        for word in words_list:
            set_words.add(word)
    return set_words

def _get_total_words_value(unigram_file):
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

def build_prefix_set(set_words):
    valid_prefixes = set()
    for word in set_words:
        for i in range(len(word) + 1):
            valid_prefixes.add(word[:i])
    return valid_prefixes


def suggestion(query):
    possible_suggestions = []
    query_split = query.split()
    query_len = len(query_split)

    if query_len == 1:
        # TODO new error model
        if query in valid_prex:
            for word in set_words:
                if query in word[:len(query)]:
                    possible_suggestions.append(word)
        else:
            possible_suggestions = [c[0] for c in correction(query)]

        return get_best_n_suggestions(possible_suggestions, 10)

    else:
        last_word = query_split[-1:][0]
        if last_word in valid_prex:
            for word in set_words:
                if last_word in word[:len(last_word)]:
                    possible_suggestions.append(word)
        else:
            last_two_words = ' '.join(query_split[-2:])
            possible_suggestions = [c[0] for c in correction(last_two_words)]

        return get_best_n_bigram_suggestions(possible_suggestions, query, 10)


# set_words, WORDS, N = _get_total_words_value(UNIGRAM_CATALOGUE)
N = 10
set_words = _generate_catalogue_dictionary(NGRAMS_CATALOGUE)
valid_prex = build_prefix_set(set_words)

def get_best_n_suggestions(suggestions, n):
    digging_suggestions = dict()
    for suggestion in suggestions:
        for w1 in product_model:
            value = generic_model[w1][suggestion]
            if value > 0:
                if w1 is not None:
                    key = w1 + " " + suggestion
                else:
                    key = suggestion
                digging_suggestions[key] = value

        for w2 in product_model[suggestion]:
            digging_suggestions[suggestion + " " + w2] = generic_model[suggestion][w2]

    best_suggestions = sorted(digging_suggestions.items(),
                   key=lambda kv: kv[1],
                   reverse=True)
    return best_suggestions[:n]

def get_best_n_bigram_suggestions(suggestions, query, n):
    w1 = query.split()[-2:-1][0]
    digging_suggestions = dict()
    for suggestion in suggestions:
        value = generic_model[w1][suggestion]
        if value > 0:
            if w1 is not None:
                key = w1 + " " + suggestion
            else:
                key = suggestion
            digging_suggestions[key] = value

    best_suggestions = sorted(digging_suggestions.items(),
                   key=lambda kv: kv[1],
                   reverse=True)
    return best_suggestions[:n]

def P(word, N=N):
    return WORDS[word] / N

def correction(query):
    words = query.split()
    if len(words) == 1:
        candidates_list = list(candidates(query))

        # check if any known candidate exist
        if len(candidates_list) == 1 and candidates_list[0] == query:
            candidates_list = []
            # genereation on prefix candidates
            p_candidates = prefix_candidates(query)
            for p_c in p_candidates:
                for w in set_words:
                    if p_c in w[:len(p_c)]:
                        candidates_list.append(w)

        bigram_candidates = dict()
        for c in candidates_list:
            for w1 in product_model:
                bigram_candidates[w1 + " " + c] = product_model[w1][c]
            # bigram_candidates[c] = product_model[None][c] # * (WORDS[c] / N)
            for w2 in product_model[c]:
                bigram_candidates[c + " " + w2] = product_model[c][w2]
        return sorted(bigram_candidates.items(), key=lambda x: x[1],
                          reverse=True)
    else:
        first_word = words[-2:-1][0]
        last_word = words[-1:][0]
        candidates_list = list(candidates(last_word))

        if len(candidates_list) == 1 and candidates_list[0] == last_word:
            candidates_list = []
            p_candidates = prefix_candidates(last_word)
            for p_c in p_candidates:
                for w in set_words:
                    if p_c in w[:len(p_c)]:
                        candidates_list.append(w)

        bigram_candidates = dict()
        for c in candidates_list:
            bigram_candidates[c] = product_model[first_word][c] * (WORDS[c] / N)

        return sorted(bigram_candidates.items(), key=lambda x:x[1],reverse=True)


def candidates(word):
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def known(words):
    return set(w for w in words if w in set_words)

def prefix_candidates(word):
    return prefix_known([word]) or \
           prefix_known(edits1(word)) or \
           prefix_known(edits2(word)) or \
           [word]

def prefix_known(words):
    return set(w for w in words if w in valid_prex)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

if __name__ == "__main__":

    # print(suggestion("acciaio zinxat"))
    # print(suggestion("acciaio zinxa"))
    # print(suggestion("zinxa"))
    #
    #
    #
    # # print(suggestion("ac"))
    # # print(suggestion("accia"))
    # # print(suggestion("acxi"))
    # # print(correction("acxi"))
    # # print(correction("acxia"))
    # # print(correction("acxiai"))
    # # print(correction("acxiaio"))
    #
    #
    # print("### Acciaio ###")
    # print(suggestion("a"))
    # print(suggestion("ac"))
    # print(suggestion("acc"))
    # print(suggestion("acco"))
    # print(suggestion("accoao"))
    # print(suggestion("accoaoo"))
    #
    # print("### Anelli ###")
    # print(suggestion("a"))
    # print(suggestion("an"))
    # print(suggestion("anr"))
    # print(suggestion("anrl"))
    # print(suggestion("anrll"))
    # print(suggestion("anrlli"))
    # print(suggestion("anelli t"))
    # print(suggestion("anelli te"))
    #
    #
    # #TODO need fix
    # print(suggestion("anello te"))

    print(suggestion("contatori"))