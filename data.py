import nltk
import numpy as np
from nltk.corpus import brown, stopwords
from collections import Counter

nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)


def load_data(max_vocab=500, max_sentences=2000):
    stop_words = set(stopwords.words('english'))
    raw_sentences = brown.sents()[:max_sentences]

    word_counts = Counter()
    for sent in raw_sentences:
        tokens = [w.lower() for w in sent
                  if w.isalpha() and w.lower() not in stop_words]
        if len(tokens) >= 2:
            word_counts.update(tokens)

    vocab_set = set(word for word, _ in word_counts.most_common(max_vocab))

    training_data = []
    for sent in raw_sentences:
        tokens = [w.lower() for w in sent
                  if w.isalpha() and w.lower() not in stop_words
                  and w.lower() in vocab_set]
        if len(tokens) >= 2:
            training_data.append(tokens)

    print(f"Loaded {len(training_data)} sentences | dictionary: {len(vocab_set)} words")
    return training_data, word_counts


def build_vocab(training_data):
    counts = Counter(word for sent in training_data for word in sent)
    vocab_list = sorted(counts.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    return vocab_list, word_to_idx, counts


def build_noise_distribution(counts, vocab_list, power=0.75):
    freqs = np.array([counts[w] for w in vocab_list], dtype=float)
    freqs = freqs ** power
    freqs /= freqs.sum()
    return freqs


def generate_skipgram_pairs(training_data, word_to_idx, window_size):
    pairs = []
    for sentence in training_data:
        indices = [word_to_idx[w] for w in sentence]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs