# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:02:17 2022

@author: https://github.com/facebookresearch/pytorch_GAN_zoo
"""

import numpy as np
import os
import torch
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm 

import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
#-----Diff Code----------------------------------------------------
use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-512',
#                        pretrained=True, useGPU=use_gpu)

# this model outputs 256 x 256 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)

#-----------------------------------------------------------------------

import torch.nn.functional as F

def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    model_name (string): if pretrained, load one of the following models
    celebaHQ-256, celebaHQ-512, DTD, celeba, cifar10. Default is celebaHQ.
    """
    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs.get('useGPU', True),
                 storeAVG=True,
                 **kwargs['config'])

    # Web Download----------------------------------------------------------------------------------------------------------------
    # checkpoint = {"celebAHQ-256": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ_s6_i80000-6196db68.pth',
    #               "celebAHQ-512": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth',
    #               "DTD": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/testDTD_s5_i96000-04efa39f.pth',
    #               "celeba": "https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s5_i83000-2b0acc76.pth"}
    # if pretrained:
    #     if "model_name" in kwargs:
    #         if kwargs["model_name"] not in checkpoint.keys():
    #             raise ValueError("model_name should be in "
    #                                 + str(checkpoint.keys()))
    #     else:
    #         print("Loading default model : celebaHQ-256")
    #         kwargs["model_name"] = "celebAHQ-256"
    #     state_dict = model_zoo.load_url(checkpoint[kwargs["model_name"]],
    #                                     map_location='cpu')
    #     model.load_state_dict(state_dict)
    #-------------------------------------------------------------------------------------------------------------------------------
    
    #Offline Download of pre-trained model --------------------------------------------------------------------------------------------------------------
    if pretrained and loadFromIndex==None:
        if "model_name" in kwargs:
            if kwargs["model_name"]=="celebAHQ-256":
                PATH="./saved_pth/celebaHQ_s6_i80000-6196db68.pth"
                model.load_state_dict(torch.load(PATH))
            else: 
                print ("Error: path not available")
    else:
        model.load_state_dict(torch.load(modelPath))
    
    return model
        
        

#1) Load pre-trained model
loadFromIndex=None
# Naive Transfer loading
stateDictNum=200
pathPrefix="../output/attempt5_Gender/"
modelPath=pathPrefix+"state_dict_%i.pth"%stateDictNum
model=PGAN(pretrained=False, model_name='celebAHQ-256', loadFrom=modelPath )
    

#LOad Fixed Noise
num_images = 10000
batchSize=10
noise, _ = model.buildNoiseData(num_images)

#Load old checkpoint
if loadFromIndex==None:
    start=0
else:
    start=loadFromIndex

out=[]
for i in tqdm(range(int(num_images/batchSize))):
    with torch.no_grad():
        generated_images = model.test(noise[i*batchSize:(i+1)*batchSize,:])
        out.append(np.array(generated_images))

out=np.vstack(np.array(out))

#Covert generated Images to [0,1] range
transformList = [Transforms.Normalize((-1,-1,-1), (2,2,2))]
transforms = Transforms.Compose(transformList)
out=transforms(torch.from_numpy(out).clamp(min=-1, max=1))
torch.save(out,pathPrefix+"generatedImage_state_dict_%i.pth"%stateDictNum)

