import nltk


def tokenize(sentence):
    nltk.download("punkt")
    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower)


def bag_of_words(tokenized_sentence, all_words):
    pass
