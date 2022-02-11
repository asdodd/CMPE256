# Import packages
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import cv2
import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm

# Path to Images
folder = "/Volumes/Mac Data/Carla and John Visiting"
imageFileType = '/*.JPG'

# Define Euclidian distance
def euclidian_distance(x,y):
  eucl_dist = np.linalg.norm(x - y)
  return eucl_dist

# Define Feature Extrator Class
class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


# Create FeatureExtractor Object
model = FeatureExtractor()

# Init Lists
image_list = []
featureList = []

# Search through folder for images
for filename in tqdm(glob.glob(folder + imageFileType)): 
    im=Image.open(filename)
	# Save extracted features to list
    featureList.append(model.extract(im))
	# Save fileNames to List
    image_list.append(filename)

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
im1 = Image.open(image_list[0])
im2 = Image.open(image_list[minIndex])
# Horizontal Stack the Two images to display at once
finalImage = np.hstack([im1,im2])
# Convert the color scheme from BGR to RGB 
# (OpenCV Assumes BGR which will display the colors incorrectly)
finalImage = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
# Show image and wait for user to press a key to end program
cv2.imshow("final",finalImage)
cv2.waitKey()
