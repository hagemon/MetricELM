from elm import MetricELM
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Dataset
iris = datasets.load_iris()
x_data, y_data = iris['data'], iris['target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)


max_recall = np.zeros([4])
max_c = None
for x in [x for x in range(-20, 20)]:
    elm = MetricELM(hidden_num=10, c=2**x)
    elm.fit(x_train, y_train)
    recall = elm.validation(x_test, y_test)
    print("c=", x, "R@{1,2,4,8}:", np.around(recall, decimals=3))
    if recall[0] > max_recall[0]:
        max_recall = recall
        max_c = x
print('best:')
print("c=", max_c, "R@{1,2,4,8}:", np.around(max_recall, decimals=3))
