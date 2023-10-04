import nltk
from nltk.stem.porter import PorterStemmer

# nltk.download("punkt")
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass

# Testing the tokenize function
a = "I want to learn a new bass riff"
print(a)
a = tokenize(a)
print(a)

# Testing the stem function
a = "I want to learn a new bass riff"
print(a)
a = stem(a)
print(a)