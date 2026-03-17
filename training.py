from config import MAX_VOCAB, MAX_SENTENCES, EPOCHS, EMBEDDING_DIM, WINDOW_SIZE, LEARNING_RATE, NEG_SAMPLES
from data import load_data, build_vocab, build_noise_distribution, generate_skipgram_pairs
from word2vec import Word2Vec

training_data, word_counts = load_data(max_vocab=MAX_VOCAB, max_sentences=MAX_SENTENCES)

vocab_list, word_to_idx, counts = build_vocab(training_data)
noise_dist = build_noise_distribution(counts, vocab_list)
pairs = generate_skipgram_pairs(training_data, word_to_idx, WINDOW_SIZE)

model = Word2Vec(
    vocab_size=len(vocab_list),
    embedding_dim=EMBEDDING_DIM,
    learning_rate=LEARNING_RATE,
    neg_samples=NEG_SAMPLES,
    noise_dist=noise_dist,
)

model.train(pairs, EPOCHS)

test_words = ["money", "city", "police"]
for word in test_words:
    if word in word_to_idx:
        result = model.predict(word_to_idx[word], word_to_idx, vocab_list, 5)
        print(f"\nWords most associated with word '{word}': {result}")