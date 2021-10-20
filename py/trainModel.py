import matplotlib
matplotlib.use("Agg")


#### Need to set up import of dataset loaders######
# from loadMnist import *
# from "./load_AZ.py" import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Setting up the argument parser and adding arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True,help="path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output training history file")
args = vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# Loading in the datasets
print("...loading datasets...")
(azData, azLabels) = load_az(args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

# Letter labels are 0-25 for letters in alphabet
# since we are combining with the digits, we shift
# the alphabet up 10 so there are no repeats
azLabels += 10

# Combine digits and az datasets
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# All images from both datasets are 28x28 pixels
# ResNet architechture requires 32x32 pixel images
# so resizing 28x28 to 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data.np.expand_dims(data, axis=-1)
data /= 255.0

# Convert the labels from ints to vectors
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = []

# Loop over all the classes and calculate the weights
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Split the data into test and training datasets
# test_size specifies test set to be 20%
(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Image generator for image augmentation
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

# Initialize and compile deep neural network
print("...compiling model...")
opt = SGD(ls=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Training the network
print("...Training the network...")
H = model.fit(
    aug.flow(Xtrain, Ytrain, batch_size=BS),
    validation_data=(Xtest, Ytest),
    steps_per_epoch=len(Xtrain) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1
)

# Define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Evaluating the neural network performance
print("...evaluating performance...")
predictions = model.predict(Xtest, batch_size=BS)
print(classification_report(Ytest.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames
                            ))
