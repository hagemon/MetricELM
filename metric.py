from elm import MetricELM
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x_data, y_data = iris['data'], iris['target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)

elm = MetricELM(hidden_num=10)
elm.fit(x_train, y_train)
pred = elm.predict(x_test)
