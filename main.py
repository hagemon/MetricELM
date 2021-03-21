from elm import MetricELM
import numpy as np
from datasets import digits, iris
from visualize import Visualizer


# Dataset
x_train, x_test, y_train, y_test = iris()
# x_train, x_test, y_train, y_test = digits()


max_recall = np.zeros([4])
max_c = None
fig_x = [x for x in range(-20, 20)]
vis = Visualizer(num_x=len(fig_x))

for i, x in enumerate(fig_x):
    elm = MetricELM(hidden_num=4, c=2**x)
    elm.fit(x_train, y_train)
    recall = elm.validation(x_test, y_test)

    # visualization
    print("c=", x, "R@{1,2,4,8}:", np.around(recall, decimals=3))
    vis.set(i, recall)

    # record max
    if recall[0] > max_recall[0]:
        max_recall = recall
        max_c = x

print('best:')
print("c=", max_c, "R@{1,2,4,8}:", np.around(max_recall, decimals=3))

vis.plot(
    x=fig_x,
    max_c=max_c,
    max_recall=max_recall
)

vis.show()
# vis.save('resources/iris.png')

