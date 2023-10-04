import json

from nltk_utils import bag_of_words, stem, tokenize

with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    # for each single intent
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", "*", "'", ".", ","]

print(all_words)
