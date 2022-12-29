# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 22:59:25 2022

@author: Chris
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms,models
import matplotlib.pyplot as plt
from tqdm import tqdm


# from clf_models import BasicBlock, build_model, Net
from utils import save_checkpoint
from dataset_splits import (
    build_even_celeba_classification_dataset,
	build_celeba_classification_dataset,
	build_multi_celeba_classification_datset,
    build_even_celeba_classification_dataset_name
)

def resnet18():
    model = models.resnet18()
    # model.classifier[1] = nn.Linear(model.last_channel, 2) #To be determined
    model.fc= nn.Linear(512, 2)
    model.to(device)
    return model

def mobilenet():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

def sanityCheck(dataLoader,predArray):
    count=0
    for i, data in tqdm(enumerate(dataLoader)):
        for j in range(len(data[0])):
            plt.imsave("./SanityCheck/%i_label_%i.jpg"%(count,predArray[count]),np.array(data[0][j].permute(1,2,0).clamp(0,1)))
            count+=1
    
    
if __name__ == "__main__":

    device = torch.device('cuda')    
    model = resnet18()
    stateDictPath="./results/attr_clf/Black_hair/model_best.pth.tar"
    model.load_state_dict(torch.load(stateDictPath)['state_dict'])
    
    dataPath="../output/attempt5_Blackhair/generatedImage_state_dict_60.pth"
    x=torch.load(dataPath)
    
    #Preprocess image to follow the classifier's training parameters
    preprocess = transforms.Compose([transforms.Resize(224)])
    x=preprocess(x.clamp(0,1)) 
    dataSet=torch.utils.data.TensorDataset(torch.tensor(x))
    dataLoader=torch.utils.data.DataLoader(dataSet,batch_size=64)
    
    #Pass model to cuda
    # model.to(device)

    predArray=[]
    for i, data in tqdm(enumerate(dataLoader)):
        logits = model(data[0].to(device))
        probas=F.softmax(logits)
        _, pred = torch.max(probas, 1)
        predArray.append(pred.detach().cpu())
        
    predArray=np.concatenate(predArray)
    
    label0Count=len(predArray)-sum(predArray)
    label1Count=sum(predArray)
    phat=[sum(predArray)/len(predArray), label0Count/len(predArray)]
    FD=np.sqrt(np.sum((np.array([0.5,0.5])-np.array(phat))**2))
    print ("p-hat distribution: "+str(phat) + "with FD: "+ str(FD) )

    # sanityCheck(dataLoader,predArray)
    
    
    # def test(epoch, loader):
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     num_examples = 0
    #     with torch.no_grad():
    #         for data, target in loader:
    #             data, target = preprocess(data).to(device), target.to(device)
    #             data = data.float() / 255.
    #             target = target.long()

    #             # run through model
    #             # logits, probas = model(data)
    #             logits= model(data)
    #             probas=F.softmax(logits)
    #             # print (logits[0])
    #             # print (target[0])
    #             test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
    #             _, pred = torch.max(probas, 1)
    #             num_examples += target.size(0)
    #             correct += (pred == target).sum()

