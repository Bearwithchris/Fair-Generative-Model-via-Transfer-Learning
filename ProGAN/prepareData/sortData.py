# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:39:20 2022

@author: Chris
"""
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch

#Preprocessing attribute list--------------------------------------------------------------
path="../../"
labelsDf=pd.read_csv(os.path.join(path, "CelebAMask-HQ-attribute-anno.csv"))
keys=list(labelsDf.keys())

#Variable to be adjusted to the SA
selectedKey=keys[40] #21=Gender
#Select Split mode (perc/min)
mode="perc"

images=list(labelsDf[keys[0]])
filteredLabels=list(labelsDf[selectedKey])
print ("Orignal labelRatio=[%f,%f] with sampleCount=[%i,%i]"%(len(np.where(np.array(filteredLabels)==0)[0])/len(filteredLabels)
                                                              ,len(np.where(np.array(filteredLabels)==1)[0])/len(filteredLabels)
                                                              ,len(np.where(np.array(filteredLabels)==0)[0])
                                                              ,len(np.where(np.array(filteredLabels)==1)[0])))  
#maxPerLabel=min(len(np.where(np.array(filteredLabels)==0)[0]),len(np.where(np.array(filteredLabels)==1)[0]))

if mode=="perc":
    totalSamples=30000#50000 #CelebA-HQ
    perc=0.05
    numDref=totalSamples*perc   
    label0=np.where(np.array(filteredLabels)==0)[0]
    label1=np.where(np.array(filteredLabels)==1)[0]
    label0=np.where(np.array(filteredLabels)==0)[0][len(label0)-int(numDref/2):len(label0)]
    label1=np.where(np.array(filteredLabels)==1)[0][len(label1)-int(numDref/2):len(label1)]
    #Select images by index
    index=np.concatenate((label0,label1))
    #Load images
    images=np.array([int(i.replace(".jpg","")) for i in images])[index]
    filteredLabels=np.array(filteredLabels)[index]
else: #min
    split={"train":0.8,"test":0.1,"val":0.1}
    label0=np.where(np.array(filteredLabels)==0)[0]
    label1=np.where(np.array(filteredLabels)==1)[0]
    numDref=min(len(label0),len(label1))
    label0=label0[0:numDref]
    label1=label1[0:numDref]
    
    #Select images by index
    trainEndIndex=int(numDref*split["train"])
    testEndIndex=int(numDref*split["train"]+numDref*split["test"])
    valEndIndex=int(numDref*split["train"]+numDref*split["test"]+numDref*split["val"])
    #Index of selected samples
    trainIndex=np.concatenate((label0[0:trainEndIndex],label1[0:trainEndIndex]))
    testIndex=np.concatenate((label0[trainEndIndex:testEndIndex],label1[trainEndIndex:testEndIndex]))
    valIndex=np.concatenate((label0[testEndIndex:valEndIndex],label1[testEndIndex:valEndIndex]))
    #Load images
    images=np.array([int(i.replace(".jpg","")) for i in images])
    filteredLabels=np.array(filteredLabels)
    
    #SepearteImages
    imagesTrain=images[trainIndex]
    filteredLabelsTrain=filteredLabels[trainIndex]
    imagesTest=images[testIndex]
    filteredLabelsTest=filteredLabels[testIndex]
    imagesVal=images[valIndex]
    filteredLabelsVal=filteredLabels[valIndex]
    
    

#Preprocessing data list from file--------------------------------------------------------------
dataPath="../../celebA-HQ/data256x256"

outPathPrefix="../../Data"
if mode=="perc":
    dataList=np.array(os.listdir(dataPath))[images]
    outImages=[]
    #Opening the samples by index
    for i in dataList:
        im = np.array(Image.open(os.path.join(dataPath, i)))
        outImages.append(im)
        
        
    outImages=np.array(outImages) #Samples (need to process between -1 to 1)
    outLabels=filteredLabels #Labels 
    
    outPathData=os.path.join(outPathPrefix,"CelebAHQ_%f_%s_data.pt"%(perc,selectedKey))
    outPathLabels=os.path.join(outPathPrefix,"CelebAHQ_%f_%s_Labels.pt"%(perc,selectedKey))
    torch.save(torch.from_numpy(np.moveaxis(outImages,3,1)), outPathData)
    torch.save(torch.from_numpy(outLabels), outPathLabels)
    
else: #Training data
    dataListTrain=np.array(os.listdir(dataPath))[imagesTrain]
    dataListTest=np.array(os.listdir(dataPath))[imagesTest]
    dataListVal=np.array(os.listdir(dataPath))[imagesVal]
    outImagesTrain=[]
    outImagesTest=[]
    outImagesVal=[]
    for i in dataListTrain:
        im = np.array(Image.open(os.path.join(dataPath, i)))
        outImagesTrain.append(im)
    for i in dataListTest:
        im = np.array(Image.open(os.path.join(dataPath, i)))
        outImagesTest.append(im)
    for i in dataListVal:
        im = np.array(Image.open(os.path.join(dataPath, i)))
        outImagesVal.append(im)
        
    outImagesTrain=np.array(outImagesTrain)
    outImagesTest=np.array(outImagesTest)
    outImagesVal=np.array(outImagesVal)
    
    #OutTrain
    outPathData=os.path.join(outPathPrefix,"CelebAHQ_even_%s_data_Train"%(selectedKey))
    outPathLabels=os.path.join(outPathPrefix,"CelebAHQ_even_%s_Labels_Train"%(selectedKey))
    np.savez(outPathData, x=torch.from_numpy(np.moveaxis(outImagesTrain,3,1)))
    np.savez(outPathLabels, x=torch.from_numpy(filteredLabelsTrain))
    #Problem with saving as .pth when file size becomes too big
    # torch.save(torch.from_numpy(np.moveaxis(outImagesTrain,3,1)), outPathData)
    # torch.save(torch.from_numpy(filteredLabelsTrain), outPathLabels)
    
    #OutTest
    outPathData=os.path.join(outPathPrefix,"CelebAHQ_even_%s_data_Test"%(selectedKey))
    outPathLabels=os.path.join(outPathPrefix,"CelebAHQ_even_%s_Labels_Test"%(selectedKey))
    np.savez(outPathData,x=torch.from_numpy(np.moveaxis(outImagesTest,3,1)))
    np.savez(outPathLabels,x=torch.from_numpy(filteredLabelsTest))
    #Problem with saving as .pth when file size becomes too big
    # torch.save(torch.from_numpy(np.moveaxis(outImagesTest,3,1)), outPathData)
    # torch.save(torch.from_numpy(filteredLabelsTest), outPathLabels)
    
    #OutVal
    outPathData=os.path.join(outPathPrefix,"CelebAHQ_even_%s_data_Val"%(selectedKey))
    outPathLabels=os.path.join(outPathPrefix,"CelebAHQ_even_%s_Labels_Val"%(selectedKey))
    np.savez(outPathData,x=torch.from_numpy(np.moveaxis(outImagesVal,3,1)))
    np.savez(outPathLabels,x=torch.from_numpy(filteredLabelsVal))
    #Problem with saving as .pth when file size becomes too big
    # torch.save(torch.from_numpy(np.moveaxis(outImagesVal,3,1)), outPathData)
    # torch.save(torch.from_numpy(filteredLabelsVal), outPathLabels)

