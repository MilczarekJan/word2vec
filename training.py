from config import MAX_VOCAB, MAX_SENTENCES, EPOCHS, EMBEDDING_DIM, WINDOW_SIZE, LEARNING_RATE
from data import load_data, prepare_data_for_training
from word2vec import Word2Vec

training_data = load_data(max_vocab=MAX_VOCAB, max_sentences=MAX_SENTENCES)

w2v = Word2Vec(embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE, learning_rate=LEARNING_RATE)
prepare_data_for_training(training_data, w2v)
w2v.train(EPOCHS)

test_words = ["money", "city"]
for word in test_words:
    result = w2v.predict(word, 5)
    if result:
        print(f"\nThe words most associated with '{word}': {result}")