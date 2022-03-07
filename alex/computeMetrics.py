# Import packages
import os

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial import distance

import cv2
from PIL import Image
import random

# Put in path to feature and image list files
baseFilePath = '/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/cnnData/'

for cnnName in ['xception', 'resnet', 'VGG16', 'VGG19', 'inception', 'mobile']:
	for simMeas in [distance.cityblock, distance.euclidean, distance.cosine]:
		featureFile = baseFilePath + 'featuresPCA_' + cnnName + '.npy'
		imageFile = baseFilePath + 'imagesPCA_' + cnnName + '.npy'


		print(f'CNN: {cnnName}, simMeasure: {simMeas}')

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

		imageToCategory = {}
		for i in range(len(image_list)):
			paths = image_list[i].split('/')
			imageToCategory[image_list[i]] = paths[12]


		def getClosestPerImage(image_list, imageToCategory, featureList, N=5):

			# Return the top N *closest* image_list for each image
			t0 = time.time()

			closestForImagePath = {}
			for i, image_path in tqdm(enumerate(image_list)):
				# Find category of query image
				queryCategory=imageToCategory[image_path]

				# Get N closest image_list and store in dictionary
				idx_closest = get_closest_images(i, featureList, simMeas, N)
				closestForImagePath[image_path] = idx_closest

				if i % 500 == 0:
					print("analyzing image %d / %d" % (i, len(image_list)))

			elap = time.time() - t0;
			print("Total time: %5.4f seconds." % elap)

			return closestForImagePath

		def get_closest_images(query_image_idx, pca_feat, simMeas, num_results=5):
			distances = [ simMeas(pca_feat[query_image_idx], feat) for feat in pca_feat ]
			idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
			return idx_closest

		def get_concatenated_images(indexes, thumb_height):
			thumbs = []
			for idx in indexes:
				img = image.load_img(image_list[idx])
				img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
				thumbs.append(img)
			concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
			return concat_image



		closestForImagePath= getClosestPerImage(image_list, imageToCategory, featureList, N=5)

		# # Get labels

		# # Find set the first picture as the Compare against Image
		# # Find the minDistance between features of other image_list and the first image
		# minDist = np.inf
		# edist = []
		# for idx, i in enumerate(featureList):
		# 	if idx == 0:
		# 		comp = i
		# 	else:
		# 		edist.append([idx,euclidian_distance(comp,i)])

		# edist = np.array(edist)
		# edist = sorted(edist,key=lambda x: x[1]) 
		# print(edist[0][0],edist[0][1])
		# print(int(edist[0][0]))
		# # Print the minDist and minIndex
		
		# # Open the first image and its approximate match
		# print(image_list[0]) 
		# print(image_list[int(edist[0][0])])
		# im1 = Image.open(image_list[0])
		# im2 = Image.open(image_list[int(edist[0][0])])
		# im1 = np.array(im1)
		# im2 = np.array(im2)
		# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
		# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

		# # # Show image and wait for user to press a key to end program
		# # cv2.imshow("final",im1)
		# # cv2.waitKey()
		# # cv2.imshow("final",im2)
		# # cv2.waitKey()



		def computeMetrics (y_true, y_pred):

			# Create empty object to hold results
			stats = lambda:0

			stats.mac = metrics.precision_score(y_true, y_pred, average='macro')
			stats.mic = metrics.recall_score(y_true, y_pred, average='micro')
			stats.f1 = metrics.f1_score(y_true, y_pred, average='macro')
			stats.wei = metrics.f1_score(y_true, y_pred, average='weighted')
			stats.beta = metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)

			return stats



		y_true = []
		y_pred = []

		for image_path in image_list:
			cat1 = imageToCategory[image_path]
			closest = closestForImagePath[image_path]
			for i in closest:
				y_true.append(cat1)
				y_pred.append(imageToCategory[image_list[i]])

		stats = computeMetrics(y_true, y_pred)

		print('Precision=', stats.mac)
		print('Recall=', stats.mic)
		print('F1 score=', stats.f1)
		print('Weighted F1=', stats.wei)
		print('Beta=', stats.beta)