import urllib.request
import zipfile
import os
from vocab import Vocab
import random
import torch

import urllib.request
import zipfile
import os
import time

def fetch_data():
    # url = "https://data.deepai.org/ptbdataset.zip"
    # urllib.request.urlretrieve(url, "ptbdataset.zip")
    #
    # with zipfile.ZipFile("ptbdataset.zip", "r") as z:
    #     z.extractall("ptb_data")
    #
    # with open(os.path.join("ptb_data", "ptb.train.txt")) as f:
    #     train_lines = [line.split() for line in f]
    #
    # return train_lines
    print("--------------------")

    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = "text8.zip"

    start = time.time()
    if not os.path.exists(zip_path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, zip_path)
    print(f"Zip Download Time: {time.time() - start:.2f}s")

    start = time.time()
    with zipfile.ZipFile(zip_path) as z:
        print("Extracting Zip File...")
        z.extractall(".")
    print(f"Zip Extraction Time: {time.time() - start:.2f}s")

    start = time.time()
    with open("text8") as f:
        print("Reading Text...")
        text = f.read().split()
    print(f"Text Reading Time: {time.time() - start:.2f}s")

    start = time.time()
    print("Chunking Text...")
    chunk_size = 1000
    train_lines = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Text Chunking Time: {time.time() - start:.2f}s")

    return train_lines

def subsample(train_lines, vocab, t=10**-4):
    print("--------------------")
    start = time.time()
    print("Calculating Relative Frequencies...")
    total = sum(vocab.count.values())
    relative_freq = {tok: freq/total for (tok, freq) in vocab.count.items()}
    print(f"Calculating Relative Frequencies Time: {time.time() - start:.2f}s")

    start = time.time()
    print("Calculating Subsampling Distribution...")
    keep_probs = {word: min((t/relative_freq[word])**(1/2), 1) for word in vocab.count}
    keep_probs["<unk>"] = 1
    print(f"Calculating Subsampling Distribution Time: {time.time() - start:.2f}s")

    start = time.time()
    print("Subsampling...")
    subsample = [[word for word in sentence if random.random() < keep_probs.get(word, 0)] for sentence in train_lines]
    print(f"Subsampling Time: {time.time() - start:.2f}s")

    return subsample

def get_centers_and_contexts(corpus, k=5): # k is maximum context window size
    print("--------------------")
    centers = []
    contexts = []

    timer_start = time.time()
    print("Creating Center-Context Pairs...")
    for sentence in corpus:
        if len(sentence) < 2:
            continue

        for word_index in range(len(sentence)):
            window_size = random.randint(1, k)

            start = max(0, word_index - window_size)
            end = min(len(sentence), word_index + window_size + 1)
            for i in range(start, end):
                if i == word_index:
                    continue
                centers.append(sentence[word_index])
                contexts.append(sentence[i])

    centers = torch.tensor(centers)
    contexts = torch.tensor(contexts)
    print(f"Center-Context Pais Creation Time: {time.time() - timer_start:.2f}s")

    return centers, contexts

def get_negatives(contexts, vocab, n=5): # n is the number of negatives per positive context word
    print("--------------------")
    start = time.time()
    print("Calculating Negatives Distribution...")
    words = list(vocab.count.keys())
    sampling_weights = torch.tensor([vocab.count[word]**(3/4) for word in words])
    print(f"Calculating Negatives Distribution Time: {time.time() - start:.2f}s")

    start = time.time()
    print("Choosing Negatives...")
    negatives_flat = torch.multinomial(sampling_weights, n*len(contexts), replacement=True)
    print(f"Choosing Negatives Time: {time.time() - start:.2f}s")

    start = time.time()
    print("Unflattening Negative List...")
    negatives = negatives_flat.reshape((-1, n))
    print(f"Unflattening Time: {time.time() - start:.2f}s")

    return negatives

def preprocess_data():
    train_lines = fetch_data()
    vocab = Vocab(train_lines)
    subsampled = subsample(train_lines, vocab)
    corpus = vocab[subsampled] # transforms tokens into their indices
    centers, contexts = get_centers_and_contexts(corpus)
    negatives = get_negatives(contexts, vocab)

    start = time.time()
    print("Concatenating Contexts and Negatives...")
    contexts_and_negatives = torch.cat([contexts.unsqueeze(dim=1), negatives], dim=1)
    print(f"Concatenating Contexts and Negatives Time: {time.time() - start:.2f}s")

    data = (centers, contexts_and_negatives)
    return vocab, data







