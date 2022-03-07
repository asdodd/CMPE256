# Import packages
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as pp_resnet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mobile
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as pp_xception
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import os
from sklearn.decomposition import PCA

# Define Euclidian distance
def euclidian_distance(x,y):
  eucl_dist = np.linalg.norm(x - y)
  return eucl_dist

# Define Feature Extrator Class
class FeatureExtractor:
    def __init__(self,cnnName,cnn,ppFunc):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = cnn(weights='imagenet')
        if any(c in cnnName for c in ('VGG16', 'VGG19')):
            finalLayer = 'fc2'
        elif any(c in cnnName for c in ('resnet', 'inception','xception')):
            finalLayer = 'avg_pool'
        elif cnnName in 'mobilenet':
            finalLayer = 'global_average_pooling2d'
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(finalLayer).output)
        self.preprocess = ppFunc
        self.cnnName = cnnName

    def extract(self, img):
        # Resize the image
        img = tf.io.read_file(img)
        # Convert the image color space
        # Reformat the image
        img = tf.io.decode_image(img,channels=3)
        if any(c in self.cnnName for c in ('VGG16', 'VGG19','resnet','mobile')):
            img = tf.image.resize(img,[224,224])
        elif any(c in self.cnnName for c in ('inception','xception')):
            img = tf.image.resize(img,[299,299])
        img = tf.expand_dims(img,axis=0)
        img = self.preprocess(img)
        # Extract Features
        feature = self.model.predict(img)[0]
        return feature / np.linalg.norm(feature)


# Define Image Path
images_path = "/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/final/101_ObjectCategories"
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

# Load Images
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]


# Create FeatureExtractor Object with diffrent CNNs
# Set X to the CNN type desired
x=5
for i in range(x):
    i = 5
    cnnName = ["VGG16", "VGG19", "resnet", "inception", "mobile", "xception"]
    cnn = [VGG16, VGG19, ResNet152V2, InceptionV3, MobileNetV2, Xception]
    ppFunc =[pp_VGG16, pp_VGG19, pp_resnet, pp_inception, pp_mobile, pp_xception]
    cnnNameSelect = cnnName[i]
    cnnSelect = cnn[i]
    ppFuncSelect = ppFunc[i]
    model = FeatureExtractor(cnnNameSelect,cnnSelect,ppFuncSelect)

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

    # Use PCA Features

    featureList = np.array(featureList)
    pca = PCA(n_components=300)
    pca.fit(featureList)
    featureList = pca.transform(featureList)



    # Save files off to .npy files for faster comparison in the future
    fL = open('/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/featuresPCA_' + cnnName[i] +'.npy','wb')
    iL = open('/Users/alexdodd/Documents/PythonLearning/AI_Masters_SJSU/7-CMPE-256-Advanced-Data-Mining/Project/CMPE256/alex/2/imagesPCA_' + cnnName[i] +'.npy','wb')
    np.save(fL,featureList)
    np.save(iL,image_list)

    # Zip features and filenames together
    if_list = list(zip(featureList,image_list))
    # Shuffle List
    random.shuffle(if_list)
    # RE-separate lists
    featureList, image_list = zip(*if_list)


    # Find set the first picture as the Compare against Image
    # Find the minDistance between features of other images and the first image
    # Set minDist as INF
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

    # Open image
    im1 = Image.open(image_list[0])
    im2 = Image.open(image_list[minIndex])
    im1 = np.array(im1)
    im2 = np.array(im2)

    # Convert to proper color
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # # Show image and wait for user to press a key to end program
    # cv2.imshow("final",im1)
    # cv2.waitKey()
    # cv2.imshow("final",im2)
    # cv2.waitKey()

