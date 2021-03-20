import numpy as np

x = np.random.randint(1, 100, [5, 5])
y = np.random.randint(0, 2, [5])
sim = np.expand_dims(y, 0) == np.expand_dims(y, 1)
d = np.expand_dims(x, 1)-np.expand_dims(x, 0)
d = abs(d.sum(-1))
print(d)
rank = np.argsort(d)
rank = rank[:, 1:]
print(rank)
print(sim)
match = np.zeros_like(rank)
for i in range(5):
    match[i] += sim[i, rank[i]]
print(match.astype(bool))
