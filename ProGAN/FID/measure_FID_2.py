# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:37:03 2022

@author: Chris
"""

import torch
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from torchvision import transforms
from keras.datasets import cifar10


# scale an array of images to a new size
def scale_images(images):
    # resize with nearest neighbor interpolation
    preprocess = transforms.Compose([transforms.Resize(299)])
    images=preprocess(images)
    # store
    return numpy.array(images.permute(0,2,3,1))
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

refPath="../../Data/CelebAHQ_FID_ref.pt"
samplePath="../output/attempt5_Blackhair/generatedImage_state_dict_60.pth"

refData=torch.load(refPath)
testData=torch.load(samplePath)*255.0

# #Transform generated data to be on the same scale [-1,1] -> [0,1]
# transformList = [transforms.Normalize((-1,-1,-1), (2,2,2))]
# transforms = transforms.Compose(transformList)
# testData=transforms(torch.from_numpy(testData).clamp(min=-1, max=1))

refData=scale_images(refData)
testData=scale_images(testData)

refData=preprocess_input(refData)
testData=preprocess_input(testData)

# calculate fid
fid = calculate_fid(model, refData, testData)
print('FID: %.3f' % fid)

