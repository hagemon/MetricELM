from elm import ELM
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

iris = datasets.load_iris()
x_data, y_data = iris['data'], iris['target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)

hit = 0
total = 0
max_acc = 0
for i in range(100):
    if (i+1) % 10 == 0:
        print(i)
    elm = ELM(hidden_num=10)
    elm.fit(x_train, LabelBinarizer().fit_transform(y_train))
    pred = elm.predict(x_test)
    res = pred == y_test
    acc = np.mean(res)
    if acc > max_acc:
        max_acc = acc
    hit += np.sum(res)
    total += len(y_test)

print(hit/total)
print(max_acc)
