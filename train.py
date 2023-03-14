import numpy as np
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize,stem,labels_set

with open('intents.json','r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    # add tags to tags[] list
    tags.append(intent['tag'])
    
    #patterns
    for pat in intent['patterns']:
        all_words.extend(tokenize(pat))
        xy.append((tokenize(pat),intent['tag']))

ignore = ['!','?','.']
all_words = [stem(i) for i in all_words if i not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"{len(xy)} patterns")
print(f"{len(tags)} tags")
print(f"{len(all_words)} unique stem words {all_words}")

# Training set
X_train = []
y_train = []

for tokenize_pattern,tag in xy:
    label = labels_set(tokenized_sentence=tokenize_pattern,words=tag)
    X_train.append(label)

    tag_index = tag.index(tags)
    y_train.append(tag_index)

X_train = np.array(X_train)
y_train = np.array(y_train)

