import matplotlib
matplotlib.use("Agg")

from models import ResNet
from dataset_loading import load_az
from dataset_loading import load_mnist_dataset
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow import saved_model
from tensorflow.lite import TFLiteConverter
from tensorflow.keras import Sequential, layers
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

# INIT_LR = 1e-1
IMAGE_SIZE = 28
EPOCHS = 20
BATCH_SIZE = 128

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
# data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
# data /= 255.0

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

print("The shape of Xtrain: " + Xtrain.shape)

# Image generator for image augmentation
data_gen = ImageDataGenerator(
    rescale=1./255.0,
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

test_data_gen = ImageDataGenerator(rescale=1./255.0)

train_generator = data_gen.flow(
    Xtrain,
    Ytrain,
    batch_size=BATCH_SIZE,
    seed=7
)
validation_generator = test_data_gen.flow(
    Xtest,
    Ytest,
    batch_size=BATCH_SIZE,
    seed=7
)

print(train_generator.class_indices)
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
    f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

base_model.trainable=False
model = Sequential([
    base_model,
    # This layer creates a convolutional kernel
    layers.Conv2D(32,3, activation = 'relu'),
    # This layer prevents the neural network from overfitting
    layers.Dropout(0.2),
    # This layer calculates the average output of each feature map
    # of the previous layer to reduce the data
    layers.GlobalAveragePooling2D(),
    # 36 = number of classes
    layers.Dense(36, activation='softmax')

])

# Initialize and compile deep neural network
print("...compiling model...")
model.compile(
    optimizer = Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the network
print("...Training the network...")
H = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(Xtrain) // BATCH_SIZE,
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
predictions = model.predict(Xtest, batch_size=BATCH_SIZE)
print(classification_report(Ytest.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames
                            ))

# save the model
print("...serializing network...")
#model.save(args["model"], save_format="h5")
saved_model.save(model, '')

converter = TFLiteConverter.from_saved_model('')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# construct and save a plot that shows training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# initialize list of output test images
images = []

# randomly select test characters
for i in np.random.choice(np.arange(0, len(Ytest)), size=(49,)):
    # classify the character
    probs = model.predict(Xtest[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    # Extract image from test data and initialize text
    # label color as green if correct
    image = (Xtest[i] * 255).astype("uint8")
    color = (0, 255, 0)

    # Label color as red if incorrect
    if prediction[0] != np.argmax(Ytest[i]):
        color = (0, 0, 255)

    # Merge the channels into one image and resize from 32 x 32
    # to 96 x 96 so we can draw the predicted label on the image
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Add the image to list of output images
    images.append(image)

# Construct build montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]

# Show the output montage
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)
