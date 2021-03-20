from elm import MetricELM
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x_data, y_data = iris['data'], iris['target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)

max_recall = np.zeros([4])
max_seed = None
max_c = None
for x in [x for x in range(-20, 20)]:
    seed = np.random.randint(0, 500)
    elm = MetricELM(hidden_num=10, c=2**x, seed=seed)
    elm.fit(x_train, y_train)
    pred = elm.predict(x_test)
    rec = elm.validation(x_test, y_test)
    if rec[0] > max_recall[0]:
        max_recall = rec
        max_seed = seed
        max_c = x
print(max_recall)
print(max_seed)
print(max_c)
# 302 -19

