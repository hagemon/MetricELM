from elm import MetricELM
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# Dataset
iris = datasets.load_iris()
x_data, y_data = iris['data'], iris['target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)


max_recall = np.zeros([4])
max_c = None
fig_x = [x for x in range(-20, 20)]
r1_value = [0 for _ in range(len(fig_x))]
r2_value = [0 for _ in range(len(fig_x))]
r4_value = [0 for _ in range(len(fig_x))]
r8_value = [0 for _ in range(len(fig_x))]

for i, x in enumerate(fig_x):
    elm = MetricELM(hidden_num=10, c=2**x)
    elm.fit(x_train, y_train)
    recall = elm.validation(x_test, y_test)

    # visualization
    print("c=", x, "R@{1,2,4,8}:", np.around(recall, decimals=3))
    r1_value[i] = recall[0]
    r2_value[i] = recall[1]
    r4_value[i] = recall[2]
    r8_value[i] = recall[3]

    # record max
    if recall[0] > max_recall[0]:
        max_recall = recall
        max_c = x

print('best:')
print("c=", max_c, "R@{1,2,4,8}:", np.around(max_recall, decimals=3))

plt.plot(fig_x, r1_value, label="R@1")
plt.plot(fig_x, r2_value, label="R@2")
plt.plot(fig_x, r4_value, label="R@4")
plt.plot(fig_x, r8_value, label="R@8")
plt.vlines(max_c, 0, max(max_recall), linestyles='dashed')
plt.xlabel("2^x")
plt.ylabel("recall")
plt.legend(loc="best")
# plt.show()
plt.savefig("resources/comparison.png")

