from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def iris():
    data = load_iris()
    x_data, y_data = data['data'], data['target']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)
    return x_train, x_test, y_train, y_test


def digits():
    data = load_digits()
    x_data, y_data = data['data'], data['target']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=11)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    digits()
