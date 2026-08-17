from model import Model
import torch
import pickle

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = Model(len(vocab))
model.load_state_dict(torch.load("word2vec_model.pt"))
model.eval()

word_vectors = model.center_embed.weight.detach()

cat_index = vocab["cat"]
cat_vector = word_vectors[cat_index]

dog_index = vocab["dog"]
dog_vector = word_vectors[dog_index]

def cosine_similarity(word_vectors, word1, word2):
    word1_index = vocab[word1]
    word2_index = vocab[word2]

    word1_vector = word_vectors[word1_index]
    word2_vector = word_vectors[word2_index]
    return torch.dot(word1_vector, word2_vector)/(word1_vector.norm() * word2_vector.norm())

def k_nearest_neighbors(vocab, word_vectors, word, k=5):
    words = vocab.count.keys()
    cosines = {partner_word: cosine_similarity(word_vectors, word, partner_word) for partner_word in words if partner_word != word}

    top_k = sorted(cosines.items(), key=lambda item: item[1], reverse=True)[:k]
    return top_k

pairs = [
    ("nine", "zero"),
    ("cat", "dog"),
    ("nine", "cat"),
    ("king", "queen"),
]
for a, b in pairs:
    print(f"{a}-{b}: {cosine_similarity(word_vectors, a, b):.3f}")