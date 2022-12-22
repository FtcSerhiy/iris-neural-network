import config
import numpy as np
import training
from sklearn import datasets
import matplotlib as plt
import random

def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = lib.predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1

    acc = correct / len(dataset)

def main():
    iris = datasets.load_iris()
    dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
    loss_arr = []
    for ep in range(config.NUM_EPOCHS):
        random.shuffle(dataset)
        training.start(dataset, loss_arr)
    accuracy = calc_accuracy()
    print("Accuracy:", accury)

    plat.plot(loss_arr)
    plt.show()

if __name__ == '__main__':
    main()
