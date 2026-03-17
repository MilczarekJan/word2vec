import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate, neg_samples, noise_dist):
        self.V = vocab_size
        self.N = embedding_dim
        self.alpha = learning_rate
        self.K = neg_samples
        self.noise_dist = noise_dist

        self.W_in  = np.random.uniform(-0.5 / self.N, 0.5 / self.N, (self.V, self.N))
        self.W_out = np.zeros((self.V, self.N))

    def _sample_negatives(self, positive_idx):
        negatives = []
        while len(negatives) < self.K:
            sample = np.random.choice(self.V, p=self.noise_dist)
            if sample != positive_idx:
                negatives.append(sample)
        return negatives

    def feed_forward(self, center_idx, context_idx):
        h = self.W_in[center_idx]

        pos_score = sigmoid(np.dot(self.W_out[context_idx], h))

        neg_indices = self._sample_negatives(context_idx)
        neg_scores = sigmoid(np.dot(self.W_out[neg_indices], h))

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1 - neg_scores + 1e-10))

        return h, pos_score, neg_indices, neg_scores, loss

    def backpropagate(self, center_idx, context_idx, h, pos_score, neg_indices, neg_scores):
        pos_grad_out = (pos_score - 1) * h
        self.W_out[context_idx] -= self.alpha * pos_grad_out

        neg_grad_out = neg_scores.reshape(-1, 1) * h
        self.W_out[neg_indices] -= self.alpha * neg_grad_out

        grad_h = (pos_score - 1) * self.W_out[context_idx]
        grad_h += np.sum(neg_scores.reshape(-1, 1) * self.W_out[neg_indices], axis=0)
        self.W_in[center_idx] -= self.alpha * grad_h

    def train_pair(self, center_idx, context_idx):
        h, pos_score, neg_indices, neg_scores, loss = self.feed_forward(center_idx, context_idx)
        self.backpropagate(center_idx, context_idx, h, pos_score, neg_indices, neg_scores)
        return loss

    def train(self, pairs, epochs):
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            np.random.shuffle(pairs)
            for center_idx, context_idx in pairs:
                total_loss += self.train_pair(center_idx, context_idx)
            self.alpha *= 1 / (1 + self.alpha * epoch)
            print(f"Epoch {epoch:3d} | loss = {total_loss:.4f}")

    def predict(self, word_idx, word_to_idx, vocab_list, n):
        h = self.W_in[word_idx]
        scores = self.W_out @ h
        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if idx != word_idx:
                results.append(vocab_list[idx])
            if len(results) >= n:
                break
        return results