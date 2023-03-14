import torch
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

st = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# in order to find the root of the word.
def stem(word):
    return st.stem(word.lower())

def labels_set(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]

    labels = np.zeros(len(words),dtype=np.float32)
    for i,w in enumerate(words):
        if w in sentence_word:
            labels[i] = 1
    return labels



# print(tokenize("Hello how are you"))
# words = ["organize", "organizes", "organizing"]
# words = [stem(w) for w in words]
# print(words)