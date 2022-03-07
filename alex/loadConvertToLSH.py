# Import packages
import cv2
import numpy as np
from PIL import Image
import random
import imagehash

# Put in path to feature and image list files
featureFile = 'features_MobileNetV2.npy'
imageFile = 'images_MobileNetV2.npy'
newFeatureFile = featureFile[:-4] + 'LSH.npy'

# Load data
fL = open(featureFile,'rb')
iL = open(imageFile,'rb')
featureList = np.load(fL)
image_list = np.load(iL)

newFeatureList = []

for i in featureList:
	feat = imagehash.average_hash(i)
	newFeatureList.append(feat)


nfL = open(newFeatureFile,'wb')
np.save(nfL,newFeatureList)
