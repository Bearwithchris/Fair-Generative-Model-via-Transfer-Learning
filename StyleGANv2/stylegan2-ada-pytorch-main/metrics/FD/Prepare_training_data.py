# -*- coding: utf-8 -*-


import argparse
import zipfile
import PIL 
import numpy as np
from tqdm import tqdm
import pickle

import torch
import copy
# parser = argparse.ArgumentParser()


# imgzip = zipfile.ZipFile("G:\AAAI23\StyleGANv2\stylegan2-ada-pytorch-main\~\datasets\BlackhairFIDSamples.zip")
# inflist = imgzip.infolist()
# file_in_zip = imgzip.namelist()

# #Load the data
# data=[]
# for f in tqdm(inflist-1):
#     ifile = imgzip.open(f)
#     img = PIL.Image.open(ifile)
#     test=np.array(img)
#     data.append(test)
    
# labels=[]
# #Load the labels
# with open("G:/AAAI23/StyleGANv2/Prepare_data/ffhq-features-dataset-master/FFHQBlackhairLabels.pkl", 'rb') as f:
#     loaded_dict = pickle.load(f)
# for label in file_in_zip:
    
attributeList=["Gender","Blackhair","Young","Moustache","Smiling"]

for i in range(len(attributeList)):
    prefix="./data/"
    outPrefix="./data/split/"
    
    split={"train":0.8,"test":0.1,"val":0.1}
    data=torch.load(prefix+"%s_FFHQ_Clf_Training_data.pt"%attributeList[i])
    labels=torch.load(prefix+"%s_FFHQ_Clf_Training_labels.pt"%attributeList[i])
    trainSamples=int(len(labels)*split['train'])
    testSamples=int(len(labels)*split['test'])
    valSamples=int(len(labels)*split['val'])
    
    
    #Identify arg of samples
    labels0Arg=np.where(labels==0)
    labels1Arg=np.where(labels==1)
    
    _data=data[:trainSamples]
    _labels=labels[:trainSamples]
    torch.save(_data , outPrefix+'{}_FFHQ_{}_64x64.pt'.format("train",attributeList[i]))
    torch.save(_labels , outPrefix+'{}_FFHQ_{}_labels_64x64.pt'.format("train",attributeList[i]))

    _data=data[trainSamples:trainSamples+testSamples]
    _labels=labels[trainSamples:trainSamples+testSamples]
    torch.save(_data , outPrefix+'{}_FFHQ_{}_64x64.pt'.format("test",attributeList[i]))
    torch.save(_labels , outPrefix+'{}_FFHQ_{}_labels_64x64.pt'.format("test",attributeList[i]))
    
    _data=data[trainSamples+testSamples:trainSamples+testSamples+valSamples]
    _labels=labels[trainSamples+testSamples:trainSamples+testSamples+valSamples] 
    torch.save(_data , outPrefix+'{}_FFHQ_{}_64x64.pt'.format("val",attributeList[i]))
    torch.save(_labels , outPrefix+'{}_FFHQ_{}_labels_64x64.pt'.format("val",attributeList[i]))

