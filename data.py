import nltk
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
    return training_data


def prepare_data_for_training(sentences, w2v):
    word_counts = {}
    for sentence in sentences:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1

    V = len(word_counts)
    vocab_list = sorted(word_counts.keys())
    vocab = {word: i for i, word in enumerate(vocab_list)}

    for sentence in sentences:
        for i, center in enumerate(sentence):
            center_vec = [0] * V
            center_vec[vocab[center]] = 1

            context_vec = [0] * V
            for j in range(i - w2v.window_size, i + w2v.window_size + 1):
                if j != i and 0 <= j < len(sentence):
                    context_vec[vocab[sentence[j]]] += 1

            if any(context_vec):
                w2v.X_train.append(center_vec)
                w2v.y_train.append(context_vec)

    w2v.initialize(V, vocab_list)
    print(f"Training examples: {len(w2v.X_train)}")