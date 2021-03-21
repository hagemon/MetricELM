from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, num_x):
        self.r1 = [0 for _ in range(num_x)]
        self.r2 = [0 for _ in range(num_x)]
        self.r4 = [0 for _ in range(num_x)]
        self.r8 = [0 for _ in range(num_x)]

    def set(self, c, recall):
        self.r1[c] = recall[0]
        self.r2[c] = recall[1]
        self.r4[c] = recall[2]
        self.r8[c] = recall[3]

    def plot(self, x, max_c, max_recall):
        plt.plot(x, self.r1, label="R@1")
        plt.plot(x, self.r2, label="R@2")
        plt.plot(x, self.r4, label="R@4")
        plt.plot(x, self.r8, label="R@8")
        plt.vlines(max_c, 0, max(max_recall), linestyles='dashed')
        plt.xlabel("2^x")
        plt.ylabel("recall")
        plt.legend(loc="best")

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save(name):
        plt.savefig(name)
