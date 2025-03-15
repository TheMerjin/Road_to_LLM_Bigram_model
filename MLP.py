from statistics import *
import torch as t

with open(r"C:\Users\Sreek\Downloads\words_alpha.txt", "r") as file:
    content = file.read().splitlines()
    """we open the words alpha.txt and make it a list of words that are strings"""
b= {}

for w in content:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram,0) + 1

b = sorted(b.items(), key = lambda v: v[1], reverse = True)

letters = t.zeros((27,27), dtype= t.int32)

chars = sorted(set("".join(content)) )+ ["."]

stoi = {v : i+1 for i,v in enumerate(chars) } 

#actual mlp
block_size = 8
X = []
Y = []
for w in content:
    context = [0] * 8
    for i in w + ".":
        X.append(context)
        Y.append(stoi[i])
        context = context[1:] + [stoi[i]]
X= t.tensor(X)
Y = t.tensor(Y)

C = t.randn((27,2))

emb = C[X]
print(emb.shape)
