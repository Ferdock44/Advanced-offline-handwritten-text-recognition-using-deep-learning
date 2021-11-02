from tensorflow.keras.datasets import mnist
import numpy as np


def load_mnist_dataset():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    data = data.reshape((data.shape[0], 28, 28))
    return (data, labels)
