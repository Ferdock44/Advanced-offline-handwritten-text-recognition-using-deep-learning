import numpy as np


def load_az():
    data = labels = []
    for row in open('./A_Z Handwritten Data/A_Z Handwritten Data.csv'):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype='uint8')
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)

        data = np.array(data, dtype='float32')
        labels = np.array(labels, dtype='int')
        return (data, labels)
