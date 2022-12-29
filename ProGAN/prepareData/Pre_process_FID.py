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
  
#Preprocessing data list from file--------------------------------------------------------------
dataPath="../../celebA-HQ/data256x256"

outPathPrefix="../../Data"
# if mode=="perc":
dataList=np.array(os.listdir(dataPath))[0:10000]
outImages=[]
#Opening the samples by index
for i in dataList:
    im = np.array(Image.open(os.path.join(dataPath, i)))
    outImages.append(im)
        
        
outImages=np.array(outImages) #Samples (need to process between -1 to 1)

outPathData=os.path.join(outPathPrefix,"CelebAHQ_FID_ref.pt")
torch.save(torch.from_numpy(np.moveaxis(outImages,3,1)), outPathData)
    



