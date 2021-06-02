# import the necessary packages
from gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

from keras.applications.inception_v3 import preprocess_input

color_maps = {
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "twilight": cv2.COLORMAP_TWILIGHT
}

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=32,
    help="size of the images used to train the model")
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-m", "--model", type=str, required=True,
    help="model to be used, e.g.: ../../models/baseline/model.h5")
ap.add_argument("-o", "--output", type=str,
    help="path to the FOLDER where the activation map will be saved. \
         If the output is not specified, the map won't be saved")
ap.add_argument("--norm", action='store_true',
    help="normalize gradients, might help when the heatmap output is full blank (no activation)")
ap.add_argument("--color", type=str, default="viridis",
    choices=["magma", "inferno", "plasma", "viridis", "cividis", "twilight"],
    help="color of the heatmap")
args = vars(ap.parse_args())

image_size=args["size"]

# load the pre-trained CNN from disk
print("[INFO] loading model...")
model = load_model(args["model"])

# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img(args["image"], target_size=(image_size, image_size))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)
i = np.argmax(preds[0])

class_names = ['Opencountry',  'coast',  'forest',  'highway',  ',inside_city', 'mountain',  'street',  'tallbuilding']
label = class_names[i]
prob = np.max(preds[0])

label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image, normalize_grads=args["norm"])

# load the original image from disk (in OpenCV format)
orig = cv2.imread(args["image"])

# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5, colormap=color_maps[args["color"]])

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    0.8, (255, 255, 255), 2)

# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)

output_file = args["output"]
if output_file is not None:
    print("Activation map saved in "+output_file+"\n")
    cv2.imwrite(output_file, output)

cv2.imshow("Output", output)
cv2.waitKey(0)
