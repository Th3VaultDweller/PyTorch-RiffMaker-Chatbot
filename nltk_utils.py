import nltk
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    nltk.download("punkt")
    stemmer = PorterStemmer()
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower)


def bag_of_words(tokenized_sentence, all_words):
    pass
