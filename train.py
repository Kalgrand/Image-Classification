# python train.py -h

import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# For execution time calculation  
from timeit import default_timer as timer

# Import the necessary packages
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import pickle
import nn

start_time = timer()

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-e", "--epochs", required=True, help="Number of epochs")
ap.add_argument("-m", "--model", required=True, help="Output model filename (without .h5 extension)")
ap.add_argument("-a", "--augmentation", required=False, action='store_true', help="Perform augmentation")
args = vars(ap.parse_args())

# Collect the input data
# Batch size
BS = 32
# Optionally resize images to the dimension of RESIZE x RESIZE.
DO_RESIZE = True
WIDTH = 28
HEIGHT = 28
# Images color depth (3 for RGB, 1 for grayscale)
IMG_DEPTH = 3
# Fraction of validation data
VALID_SPLIT = 0.2
# Data taken from the command-line arguments
EPOCHS = int(args["epochs"])
DO_AUGMENTATION = args["augmentation"]
MODEL_BASENAME = args["model"]
HIDDEN_UNITS = 77
VERBOSE = 1
ROTATION_RANGE = 30
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.2
SHEAR_RANGE = 0.2
FILL_MODE = "nearest"
# Initialize the data and labels
print("Loading images...")
data = []
labels = []
labels_text = []

# Grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed()
random.shuffle(imagePaths)

# Loop over the input images
for imagePath in imagePaths:
    # Load the image and pre-process it
    image = cv2.imread(imagePath)
    if DO_RESIZE:
        image = cv2.resize(image, (WIDTH, HEIGHT))
    image = tf.keras.preprocessing.image.img_to_array(image)
  
    # Extract the class label (numeric and text) from the image path
    dirname = imagePath.split(os.path.sep)[-2]   
    dirname_list = dirname.split("-")
    
    if dirname_list[0] != "class":
        # File not in "class-*" directory, skip to another file
        continue
        
    # Label is the number in the directory name, after first "-" 
    label = int(dirname_list[1])
    # Text label is the text in the directory name, after second "-" 
    try:
        label_text = dirname_list[2]
    except KeyError:
        label_text = int(dirname_list[1])
           
    # Store image and labels in lists
    data.append(image)
    labels.append(label)
    labels_text.append(label_text)

# Get the unique classes names
classes = np.unique(labels_text)

# Save the text labels to disk as pickle
f = open(MODEL_BASENAME+".lbl", "wb")
f.write(pickle.dumps(classes))
f.close()

# Convert labels to numpy array
labels = np.array(labels)

# Determine number of classes    
no_classes = len(classes)
    
# Scale the raw pixel intensities to the [0, 1] range
data = np.array(data, dtype="float") / 255.0

# Data partitioning (only if augmentation is enabled)
if DO_AUGMENTATION:
    # Partition the data into training and validating splits
    (train_data, valid_data, train_labels, valid_labels) = \
        train_test_split(data, labels, test_size=VALID_SPLIT)
else:
    # Data partitioning will be done automatically during training
    train_data = data
    train_labels = labels

# Convert the labels from integers to category vectors
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=no_classes)
if DO_AUGMENTATION:
    valid_labels = tf.keras.utils.to_categorical(valid_labels, num_classes=no_classes)

# Data augmentation
if DO_AUGMENTATION:
    print("Perform augmentation...")
    # Construct the image generator for data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=ROTATION_RANGE, width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE, shear_range=SHEAR_RANGE , zoom_range=ZOOM_RANGE,
        horizontal_flip=True, fill_mode="nearest")

# Initialize the model
print("Compiling model...")
#model = nn.FullyConnectedForImageClassisfication.build(width=WIDTH, height=HEIGHT, depth=IMG_DEPTH, hidden_units=HIDDEN_UNITS, classes=no_classes)
model = nn.SmallerVGGNet.build(width=WIDTH, height=HEIGHT, depth=IMG_DEPTH, classes=no_classes)
#model = nn.LeNet5.build(width=WIDTH, height=HEIGHT, depth=IMG_DEPTH, classes=no_classes)
model.summary()

# Select the loss function
if no_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
model.compile(loss=loss, optimizer="Adam", metrics=["accuracy"])

# Train the network
print("Training network...")
if DO_AUGMENTATION:
    H = model.fit(x=aug.flow(train_data, train_labels, batch_size=BS), 
        validation_data=(valid_data, valid_labels), epochs=EPOCHS, verbose=1)    
else:
    H = model.fit(x=train_data, y=train_labels, batch_size=BS,
        validation_split=VALID_SPLIT, epochs=EPOCHS, verbose=VERBOSE)

# Save model to disk
print("Saving model and plots...")
model.save(MODEL_BASENAME+".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="train_loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(MODEL_BASENAME+".png")