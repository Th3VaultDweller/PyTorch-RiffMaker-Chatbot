import json
import numpy as np
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
all_words = [stem(w) for w in all_words if w not in ignore_words]  # stemming
print(all_words)

print(f"\n")

all_words = sorted(set(all_words))  # sorted function will return a list
tags = sorted(set(tags))
print(tags)

# training data
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss

x_train=np.array(x_train)
y_train=np.array(y_train)

