import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Word2Vec:
    def __init__(self, embedding_dim=50, window_size=2, learning_rate=0.01):
        self.N = embedding_dim
        self.window_size = window_size
        self.alpha = learning_rate
        self.X_train = []
        self.y_train = []
        self.words = []
        self.word_index = {}

    def initialize(self, V, data):
        self.V = V
        self.W  = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i, word in enumerate(data):
            self.word_index[word] = i

    def feed_forward(self, X):
        self.h = np.dot(self.W.T, X).reshape(self.N, 1)
        self.u = np.dot(self.W1.T, self.h)
        self.y = softmax(self.u)
        return self.y

    def backpropagate(self, x, t):
        e = self.y - np.asarray(t).reshape(self.V, 1)
        dLdW1 = np.dot(self.h, e.T)
        X = np.array(x).reshape(self.V, 1)
        dLdW = np.dot(X, np.dot(self.W1, e).T)
        self.W1 -= self.alpha * dLdW1
        self.W  -= self.alpha * dLdW

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j], self.y_train[j])
                C = 0
                for m in range(self.V):
                    if self.y_train[j][m]:
                        self.loss += -1 * self.u[m][0]
                        C += 1
                self.loss += C * np.log(np.sum(np.exp(self.u)))
            print(f"Epoch {epoch:3d} | loss = {self.loss:.4f}")
            self.alpha *= 1 / (1 + self.alpha * epoch)

    def predict(self, word, number_of_predictions):
        if word not in self.words:
            print(f"Word '{word}' is not in the dictionary.")
            return []
        index = self.word_index[word]
        X = [0] * self.V
        X[index] = 1
        prediction = self.feed_forward(X)
        output = {prediction[i][0]: i for i in range(self.V)}
        top_words = []
        for k in sorted(output, reverse=True):
            candidate = self.words[output[k]]
            if candidate != word:
                top_words.append(candidate)
            if len(top_words) >= number_of_predictions:
                break
        return top_words