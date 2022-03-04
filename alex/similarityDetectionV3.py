# Import packages
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm
import os

# Define Euclidian distance
def euclidian_distance(x,y):
  eucl_dist = np.linalg.norm(x - y)
  return eucl_dist

# Define Feature Extrator Class
class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = MobileNetV2(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

    def extract(self, img):
        # Resize the image
        img = tf.io.read_file(img)
        # Convert the image color space
        # Reformat the image
        img = tf.io.decode_image(img,channels=3)
        img = tf.image.resize(img,[224,224])
        img = tf.expand_dims(img,axis=0)
        img = preprocess_input(img)
        # Extract Features
        feature = self.model.predict(img)[0]
        return feature / np.linalg.norm(feature)


#
images_path = "/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/101_ObjectCategories"
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]


# Create FeatureExtractor Object
model = FeatureExtractor()
# print(model.model.layers)

# Init Lists
image_list = []
featureList = []

# Search through folder for images
for im in tqdm(images):
    # im=Image.open(filename)
	# Save extracted features to list
    featureList.append(model.extract(im))
	# Save fileNames to List
    image_list.append(im)


fL = open('/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/features_MobileNetV2.npy','wb')
iL = open('/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/images_MobileNetV2.npy','wb')
np.save(fL,featureList)
np.save(iL,image_list)

# Zip features and filenames together
if_list = list(zip(featureList,image_list))
# Shuffle List
random.shuffle(if_list)
# RE-separate lists
featureList, image_list = zip(*if_list)

# Set minDist as INF
minDist = np.inf



# Find set the first picture as the Compare against Image
# Find the minDistance between features of other images and the first image
for idx, i in enumerate(featureList):
  if idx == 0:
    comp = i
  else:
    edist = euclidian_distance(comp,i)
    if edist < minDist:
      minDist = edist
      minIndex = idx

# Print the minDist and minIndex
print(minDist, minIndex)
  

# Open the first image and its approximate match
print(image_list[0])
print(image_list[minIndex])
im1 = Image.open(image_list[0])
im2 = Image.open(image_list[minIndex])
im1 = np.array(im1)
im2 = np.array(im2)
# Horizontal Stack the Two images to display at once
# finalImage = np.hstack([im1,im2])
# # Convert the color scheme from BGR to RGB  
# # (OpenCV Assumes BGR which will display the colors incorrectly)
# finalImage = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
# Show image and wait for user to press a key to end program
cv2.imshow("final",im1)
cv2.waitKey()
cv2.imshow("final",im2)
cv2.waitKey()

