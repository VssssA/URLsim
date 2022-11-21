# Data cleaning and processing can be done in multiple steps that involve
# 1. Sentence, word Tokenization
# 2. Expanding the contractions (we'll--> we will) This helps alot in many cases of similarity
# 3. Removing common/custom stop words
# 4. Removing punctuations
# 5. Stemming/Lemmatization (Not performed due to restrictions)
# 6. Building custom Vocabulary
# 7. Creating N-Grams (Only unigrams are considered here for simplicity)
# 8. Vectorization of texts to numbers to get TF-IDF features.
#
# Once the words are converted to vectors, maths should do the job for us to get similarity between the vectors.
# In this module, following 3 different and basic approaches to find the similarity are implemented:
# 1. Cosine Similarity
# 2. Jaccard Index Similarity
# 3. Euclidian Diatance Similarity
#
# As this is a reusable module, it is scaled to take muliple texts(even more than 3) at a time to compare.
# The similarity method can be chosen along with choice of removal of punctuations and stop words via input args.


# Initial Imports
import numpy as np
import re
from math import *
from collections import Counter
import os
import requests
from pathlib import Path
import nltk
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
import pymorphy2
from collections import Counter

# Data Cleaning
# Here we can add the custom stop words and punctuations to give importance to certain ones.
# I have selected a few based default text examples.
stop_words_ru = set(stopwords.words('russian'))
# stop_words_ru = set(stopwords.words('russian'))
punctuations_marks = ['.', ':', ',', '!', ';', '\'', '\"', '(', ')']


# Sentence and word Tokenization. Punctuation and stop word removal is optional that are performed along with this.
def tokenize_sentences(text, stop_char="."):
    return [x.lower() + ' ' + stop_char + '' for x in text.split(stop_char) if x != ""]


def tokenize_words_from_sent(sentences, stop_words_list=None, punctuation=None):
    if punctuation is None:
        punctuation = []
    if stop_words_list is None:
        stop_words_list = []
    return [x.lower() for sent in sentences for x in sent.split() if
            (x != "" and x not in punctuation and x not in stop_words_list)]


# Word Expansion
def deconstruction_en_words(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


# Considering all the separate texts contribute to our entire vocabulary/corpus, the vocabulary is created.
# Along with vocabulary, count of individual words in corpus is stored.
# The 2 dictionaries int_to_vocab and vocab_t_int comes in handy while vectorizing the words.
def prepare_vocab(all_words):
    word_counts = Counter([word for text in all_words for word in text])
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return word_counts, sorted_vocab, int_to_vocab, vocab_to_int


# This function gives Terf frequency for each text separately.
def get_word_tf(sorted_vocab, vocab_to_int, words):
    word_tf = np.zeros(len(sorted_vocab), dtype=float)
    for word, count in Counter(words).items():
        word_tf[vocab_to_int[word]] = count / len(words)

    return word_tf


# This function computes Inverse Document Frequency for each word in our vocabulay. Thus, all the words hav distinct
# IDF. Generally, the log of IDF i spreferred but for this simple implementation it is neglected. Can be added for
# improvements.
def get_word_idf(all_words, n_texts, sorted_vocab, vocab_to_int):
    word_idf = np.zeros(len(sorted_vocab), dtype=float)
    for word in sorted_vocab:
        n_docs = 0
        for doc in all_words:
            if word in doc:
                n_docs += 1
        word_idf[vocab_to_int[word]] = n_texts / n_docs

    return word_idf


# This functions gives the TF-IDF vectors for our corpus.
def get_tfidf_vectors(word_tf, word_idf):
    return [tf * idf for tf, idf in zip(word_tf, word_idf)]


# Helper function to compute square root rounded upto 3 decimals
def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


# This function computes the cosine angle(similarity) between given 2 input word vectors.
def get_cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


# This is the function which implements and performs all the above processes with respect to input arguments.
def get_text_similarity(texts, method="cosine", exclude_stopwords=False, exclude_punctuation=False):

    n_texts = len(texts)  # Total texts to compare. Need atleast 2.
    if n_texts < 2:
        return

    # Matrix to store similarity between each individual text with another.
    sim_matrix = [[0 if i != j else 1 for j in range(n_texts)] for i in range(n_texts)]
    # Check for exact similarity. The exact similar texts can be directly marked as 1 to avoid further computation.
    n_similar = 0
    for i in range(n_texts):
        for j in range(i + 1, n_texts):
            if j < n_texts and texts[i] == texts[j]:
                sim_matrix[i][j] = 1
                n_similar += 1
            else:
                pass
    # If all texts are similar, return
    if n_similar == n_texts - 1:
        return sim_matrix

    # Tokenize the words from texts and store in list.
    tokenized_stop_words = stop_words_ru if exclude_stopwords else []
    punctuations = punctuations_marks if exclude_punctuation else []
    all_words = [tokenize_words_from_sent(tokenize_sentences(deconstruction_en_words(text)),
                                          tokenized_stop_words, punctuations) for text in texts]

    # If other methods(word-vector based) are needed, prepare the data
    # The data is prepared assuming all the texts are part of the same vocabulary,
    # And thus there will be 'n_texts' documents in our corpus
    word_counts, sorted_vocab, int_to_vocab, vocab_to_int = prepare_vocab(all_words)
    #     print(word_counts,"\n*********************\n",sorted_vocab,"\n*********************\n",int_to_vocab,
    #           "\n*********************\n",vocab_to_int)

    # Calculate Word term frequencies
    word_tfs = [get_word_tf(sorted_vocab, vocab_to_int, words) for words in all_words]

    # Calculate Inverse Document frequencies for all words
    word_idf = get_word_idf(all_words, n_texts, sorted_vocab, vocab_to_int)

    # Create TF-IDF Vectors
    tfidf_vec = [get_tfidf_vectors(word_tf, word_idf) for word_tf in word_tfs]

    # Based on method, return the calculated similarity matrix
    if method == "cosine":
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                if j >= n_texts or sim_matrix[i][j] == 1:
                    break
                sim_matrix[i][j] = get_cosine_similarity(tfidf_vec[i], tfidf_vec[j])
        return sim_matrix


# Driver method
def main():
    def get_text(url, encoding='utf-8', to_lower=True):
        url = str(url)
        if url.startswith('http'):
            r = requests.get(url)
            if not r.ok:
                r.raise_for_status()
            return r.text.lower() if to_lower else r.text
        elif os.path.exists(url):
            with open(url, encoding=encoding) as f:
                return f.read().lower() if to_lower else f.read()
        else:
            raise Exception('parameter [url] can be either URL or a filename')

    print("Введите ссылки на текстовые документы")
    sample1 = get_text(input())
    sample2 = get_text(input())
    texts = [sample1, sample2]
    sim_matrix = get_text_similarity(texts, method="cosine")
    print(sim_matrix[0][1])


if __name__ == "__main__":
    main()
