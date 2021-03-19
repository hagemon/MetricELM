import numpy as np


class ELM:
    def __init__(self, hidden_num, activation='sigmoid'):
        self.hidden_num = hidden_num
        self.w = None
        self.b = None
        self.beta = None
        self.g = self.sigmoid
        if activation == 'sigmoid':
            self.g = self.sigmoid

    def fit(self, x, y):
        feature_size = x.shape[1]
        self.w = np.random.random([feature_size, self.hidden_num]) * 2 - 1
        self.b = np.random.random([self.hidden_num]) * 2 - 1

        hidden = np.matmul(x, self.w) + self.b
        hidden = self.g(hidden)
        self.beta = np.matmul(np.linalg.pinv(hidden), y)

    def predict(self, x):
        assert self.w is not None
        hidden = self.g(np.matmul(x, self.w) + self.b)
        pred = np.matmul(hidden, self.beta)
        return np.argmax(pred, axis=1)

    @staticmethod
    def sigmoid(x):
        return 1/(np.exp(-x)+1)


class MetricELM:
    def __init__(self, hidden_num, activation='sigmoid'):
        self.hidden_num = hidden_num
        self.w = None
        self.b = None
        self.beta = None
        self.g = self.sigmoid
        if activation == 'sigmoid':
            self.g = self.sigmoid

    def fit(self, x, y):
        feature_size = x.shape[1]
        self.w = np.random.random([feature_size, self.hidden_num]) * 2 - 1
        self.b = np.random.random([self.hidden_num]) * 2 - 1
        x = self._build_x_pairs(x)
        y = self._build_y_pairs(y)

        hidden = np.matmul(x, self.w) + self.b
        hidden = self.g(hidden)
        self.beta = np.matmul(np.linalg.pinv(hidden), y)

    def predict(self, x):
        assert self.w is not None
        x = self._build_x_pairs(x)
        hidden = self.g(np.matmul(x, self.w) + self.b)
        pred = np.matmul(hidden, self.beta)
        return np.argmax(pred, axis=1)

    @staticmethod
    def _build_x_pairs(x):
        n = len(x)
        x = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=0)
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
