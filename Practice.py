from statistics import *
import torch as t
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

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

stoi = {v : i for i,v in enumerate(chars) }

for ((x,y), z) in b:
    letters[stoi[y]][stoi[x]] = z


g = t.Generator().manual_seed(420068)
p = letters[0].float()
p = p/p.sum()
idx = t.multinomial(p, 1, replacement= True, generator=g).item()


for i in range(200):
    idx = 0  # Start at the beginning marker (".")
    word = ""
    while True:
        p = letters[idx].float()
        p = p / p.sum()
        x = t.multinomial(p, 1, replacement=True, generator=g).item()
        key = next((k for k, v in stoi.items() if v == x), None)
        
        # If the end marker is generated, stop building this word
        if key == ".":
            break
        
        word += key  # Append the generated character to the word
        idx = x      # Update idx to the newly generated character's index
    
    print(word)
