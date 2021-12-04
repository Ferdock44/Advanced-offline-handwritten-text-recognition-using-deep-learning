import matplotlib
import math
from dataset_loading import loadAZ
from dataset_loading import loadMnist
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Constant
from tensorflow import saved_model
from tensorflow import lite
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import matplotlib

matplotlib.use("Agg")

# Setting up the argument parser and adding arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output training history file")
args = vars(ap.parse_args())

IMAGE_SIZE = 128
EPOCHS = 400
BATCH_SIZE = 64
NUM_CLASSES = 36
INIT_LR = 1e-1

# # Loading in the datasets
print("...loading datasets...")
(digitsData, digitsLabels) = loadMnist.load_mnist_dataset()
(azData, azLabels) = loadAZ.load_az(args["az"])

# Letter labels are 0-25 for letters in alphabet
# since we are combining with the digits, we shift
# the alphabet up 10 so there are no repeats
azLabels += 10

# Combine digits and az datasets
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])
# data = azData
# labels = azLabels
# All images from both datasets are 28x28 pixels
# ResNet architechture requires 32x32 pixel images
# so resizing 28x28 to 32x32
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
# data *= 1.0/255.0

# data, labels = shuffle(data, labels, random_state=42)

(Xtrain, Xrem, Ytrain, Yrem) = train_test_split(data, labels, test_size=0.3, stratify=labels, shuffle=True,
                                                random_state=42)
(Xvalid, Xtest, Yvalid, Ytest) = train_test_split(Xrem, Yrem, test_size=0.5, stratify=Yrem, shuffle=True,
                                                  random_state=42)

Xtest /= 255.0

# def sample_counts(ds, dsName):
#     print("----------------------------------")
#     print(f'Number of samples per label in {dsName}: ')
#     sample_counts = {}
#     for i in range(ds.shape[0]):
#         if  ds[i] in sample_counts:
#             sample_counts[ds[i]] += 1
#         else:
#             sample_counts[ds[i]] = 1
#
#     for k,v in sample_counts.items():
#         print(k,v)
#
# sample_counts(Ytrain, "Ytrain")
# sample_counts(Yvalid, "Yvalid")
# sample_counts(Ytest, "Yvalid")

# Convert the labels from ints to vectors
le = LabelBinarizer()
_labels = le.fit_transform(labels)
counts = _labels.sum(axis=0)

# Account for skew in the labeled data
classTotals = _labels.sum(axis=0)
classWeight = []
print(classTotals.shape)
print(type(classTotals))
# Loop over all the classes and calculate the weights
for i in range(0, len(classTotals)):
    classWeight.append(classTotals.max() / classTotals[i])

# Split the data into test and training datasets
# test_size specifies test set to be 20%
# data *= 1./255.0

# Convert these numpy datasets to tensorflow datasets
print("...loading in ds_train_data...")
ds_train_data = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
print("...loading in ds_val_data...")
ds_val_data = tf.data.Dataset.from_tensor_slices((Xvalid, Yvalid))
print("...finished loading ds_val_data...")


# print(list(ds_train_data.as_numpy_iterator()))
# print(list(ds_val_data.as_numpy_iterator()))


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
print("...initializing ds_train...")
ds_train = (
    ds_train_data
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(Xtrain.shape[0])
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
)
print("...initializing ds_val...")
ds_val = (
    ds_val_data
        .map(preprocess, AUTOTUNE)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)

)
print("...beginning defining layers...")
# Image generator for image augmentation
inputs = layers.Input(shape=(28, 28, 1), name='input')

x = layers.Conv2D(24, kernel_size=(7, 7), strides=1)(inputs)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(48, kernel_size=(5, 5), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(64, kernel_size=(3, 3), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(200, activation="relu")(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.2)(x)

predications = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

# Initialize and compile deep neural network
print("...compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = Model(inputs=inputs, outputs=predications)
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
# Training the network
print("...Training the network...")
LR_DECAY = lambda epoch: 0.0001 + 0.02 * math.pow(1.0 / math.e, epoch / 3.0)
decay_callback = LearningRateScheduler(LR_DECAY, verbose=1)
H = model.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[decay_callback],
    epochs=EPOCHS,
    # class_weight=classWeight,
    verbose=1
)

# Define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Evaluating the neural network performance
print("...evaluating performance...")
predictions = model.predict(Xtest, batch_size=BATCH_SIZE)
print(classification_report(Ytest,
                            predictions.argmax(axis=1),
                            target_names=labelNames
                            ))

# save the model
print("...serializing network...")
model.save(args["model"], save_format="h5")
#saved_model.save(model, './models')

converter = lite.TFLiteConverter.from_saved_model('./models')
tflite_model = converter.convert()

with open('./models/model.tflite', 'wb') as f:
    f.write(tflite_model)

# model = tf.keras.models.load_model('./')

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
    image = (Xtest[i] * 255.0).astype("uint8")
    color = (0, 255, 0)

    # Label color as red if incorrect
    if prediction[0] != Ytest[i]:
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

# construct and save a plot that shows training history
# N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])
