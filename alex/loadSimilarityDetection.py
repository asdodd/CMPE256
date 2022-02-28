# Import packages
import cv2
import numpy as np
from PIL import Image
import random

# Put in path to feature and image list files
featureFile = '/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/cnnData/features_MobileNetV2.npy'
imageFile = '/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/cnnData/images_MobileNetV2.npy'

# Define Euclidian distance
def euclidian_distance(x,y):
  eucl_dist = np.linalg.norm(x - y)
  return eucl_dist


# Load data
fL = open(featureFile,'rb')
iL = open(imageFile,'rb')
featureList = np.load(fL)
image_list = np.load(iL)

# Zip features and filenames together
if_list = list(zip(featureList,image_list))

# Shuffle List
random.shuffle(if_list)

# Re-Separate lists
featureList, image_list = zip(*if_list)

# Find set the first picture as the Compare against Image
# Find the minDistance between features of other images and the first image
minDist = np.inf
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
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Show image and wait for user to press a key to end program
cv2.imshow("final",im1)
cv2.waitKey()
cv2.imshow("final",im2)
cv2.waitKey()
