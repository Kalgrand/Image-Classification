# python classify.py -h

# Import packages
import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import argparse
import cv2
import pickle

# Resize parameters (they should be the same as used in training)
DO_RESIZE = True
WIDTH = 28
HEIGHT = 28

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to trained model file (without extenson)")
ap.add_argument("-t", "--testset", required=True, help="Path to test images")
args = vars(ap.parse_args())

# Read labels for classes to recognize
print("Loading labels...")
f = open(args["model"]+".lbl", 'rb')
CLASS_LABELS = pickle.load(f)
f.close()

# Load the trained network
print("Loading network...")
model = tf.keras.models.load_model(args["model"]+".h5")

# Loop over images
print("Classifying...")
for image_name in os.listdir(args["testset"]):

    # Load the image
    image = cv2.imread(args["testset"]+os.path.sep+image_name)
    orig = image.copy()

    # Pre-process the image for classification
    if DO_RESIZE:
        image = cv2.resize(image, (WIDTH , HEIGHT))
    image = image.astype("float") / 255.0
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image
    prediction = list(model.predict(image)[0])
    predict = sorted(prediction, reverse=True)
    first = predict[0]
    second = predict[1]
    third = predict[2]

    # Find the winner class and the probability
    first_place = prediction.index(first)
    first_probability = round(first * 100,2)
    second_place = prediction.index(second)
    second_probability = round((second) * 100,2)
    third_place = prediction.index(third)
    third_probability = round((third) * 100,2)
    
    # Build the text label
    label_f = "{} : {}%".format(CLASS_LABELS[first_place], first_probability)
    label_s = "{} : {}%".format(CLASS_LABELS[second_place], second_probability)
    label_t = "{} : {}%".format(CLASS_LABELS[third_place], third_probability)
    
    # Draw the label on the image               1
    output_image = cv2.resize(orig, (900,900))
    cv2.putText(output_image, label_f, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(output_image, label_s, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(output_image, label_t, (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
    # Show the output image        
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)