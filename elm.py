import numpy as np


class MetricELM:
    def __init__(self, hidden_num, c, activation='sigmoid'):
        self.hidden_num = hidden_num
        self.w = None
        self.b = None
        self.c = c
        self.beta = None
        self.g = self.sigmoid
        if activation == 'sigmoid':
            self.g = self.sigmoid

    def fit(self, x, y):
        feature_size = x.shape[1]
        self.w = np.random.random([feature_size, self.hidden_num]) * 2 - 1
        self.b = np.random.random([self.hidden_num]) * 2 - 1
        y = self._build_y_pairs(y)

        hidden = np.matmul(x, self.w) + self.b
        hidden = self.g(hidden)
        hidden = self._build_x_pairs(hidden)
        self.beta = np.matmul(
            np.linalg.inv(
                np.eye(self.hidden_num) / self.c
                + np.matmul(hidden.T, hidden)
            ),
            np.matmul(hidden.T, y)
        )

    def predict(self, x):
        assert self.w is not None
        hidden = self.g(np.matmul(x, self.w) + self.b)
        hidden = np.expand_dims(hidden, axis=1) - np.expand_dims(hidden, axis=0)
        hidden = hidden ** 2
        res = np.matmul(hidden, self.beta)
        return res

    def validation(self, x, y):
        res = self.predict(x)
        sim = np.expand_dims(y, axis=1) == np.expand_dims(y, axis=0)
        rank = np.argsort(res)[:, 1:]
        recall = [0, 0, 0, 0]
        n = len(x)
        match = np.zeros_like(rank)
        for i in range(n):
            match[i] += sim[i, rank[i]]
        for i, r in enumerate([1, 2, 4, 8]):
            rec = np.sum(match[:, :r], axis=1) / r
            recall[i] = rec.mean()
        return recall

    @staticmethod
    def _build_x_pairs(x):
        n = len(x)
        x = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=0)
        x = x ** 2
        mask = (1-np.tril(np.ones(n))).astype(bool)
        x = x[mask]
        return x

    @staticmethod
    def _build_y_pairs(y):
        n = len(y)
        mask = (1 - np.tril(np.ones(n))).astype(bool)
        y = np.expand_dims(y, axis=1) == np.expand_dims(y, axis=0)
        y = y.astype(int)[mask]
        return y

    @staticmethod
    def sigmoid(x):
        return 1/(np.exp(-x)+1)
