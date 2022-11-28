# -*- coding: utf-8 -*-

import torch
import numpy as np
import re
import utils
import BigGAN as model
import matplotlib.pyplot as plt

def load_GD_dict(prefix0,suffix):
    r1=re.compile(".*conv.*")
    r2=re.compile(".*weight")
    
    #Setup config
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    
    #Generator====================================
    #Load weights
    GWeights=torch.load(prefix0+"G_"+suffix+".pth")
    GWeightsKeys=list(GWeights.keys())
    #Layers of interest
    GLOI=list(filter(r1.match,GWeightsKeys))
    GLOI=list(filter(r2.match,GLOI))
    GOut={k:v for k,v in GWeights.items() if k in GLOI}
    
    
    #Discriminator====================================
    #Load weights
    DWeights=torch.load(prefix0+"D_"+suffix+".pth")
    DWeightsKeys=list(DWeights.keys())
    #Layers of interest
    DLOI=list(filter(r1.match,DWeightsKeys))
    DLOI=list(filter(r2.match,DLOI))
    DOut={k:v for k,v in DWeights.items() if k in DLOI}
    
    return GOut,DOut

# # #10-90 Perc=0.25 (Unbalanced to balanced)
# GUnbalanced,DUnbalanced=load_GD_dict("./weights/compare/celeba_10_90_perc0.25_unbalanced_v2/","best_fid0")
# GBalanced,DBalanced=load_GD_dict("./weights/compare/celeba_10_90_perc0.25_unbalanced_v2_to_0.25_balanced_BEST_FID/","best_fid14")

# #10-90 Perc=1.0 (Unbalanced to balanced)
# GUnbalanced,DUnbalanced=load_GD_dict("./weights/compare/celeba_90_10_perc0.5_v2_unbalanced_ONLY/","best_fid1")
# GBalanced,DBalanced=load_GD_dict("./weights/compare/celeba_90_10_perc0.5_v2_unbalanced_tfl_balanced_BEST_FD/","best_fair25")

# #90-10 Perc=1.0 (Unbalanced to balanced AdaFM attempt)
GUnbalanced,DUnbalanced=load_GD_dict("./weights/Baseline_celeba_90_10_perc0.5_v2_unbalanced_to_balanced_AdaFM/","best_fid1")
GBalanced,DBalanced=load_GD_dict("./weights/Test_celeba_90_10_perc0.5_v2_unbalanced_to_balanced_AdaFM/","best_fair35")



#10-90 Perc=0.25 (unbalanced to Balanced)
# GUnbalanced,DUnbalanced=load_GD_dict("./weights/compare/celeba_90_10_perc0.1_unbalanced_Only_cont/","best_fid37")
# GBalanced,DBalanced=load_GD_dict("./weights/compare/celeba_90_10_perc0.1_unbalanced_Only_cont_to_perc0.25_balanced/","best_fid35")

#Filtering Generator Plots
GDelta={}
for k in GUnbalanced.keys():
    GDelta[k]=torch.mean(((abs(GUnbalanced[k].cpu()-GBalanced[k].cpu()))/abs(GUnbalanced[k].cpu())).flatten())

GConvDelta={}
for i in range(0,4):
    r=re.compile("blocks\.{}\.0\.conv.*".format(i))
    query=list(filter(r.match,GDelta))
    convSum=0
    count=0
    for j in query:
        if j.find("sc")==-1: #Remove short cuts
            print("Removed short cut")
            convSum+=GDelta[j]
            count+=1
    #Find the average in the block
    convSum=convSum/(count)
    GConvDelta["block{}.conv".format(i)]=convSum*100
    

#Filtering Discriminator Plots
DDelta={}
for k in DUnbalanced.keys():
    DDelta[k]=torch.mean(((abs(DUnbalanced[k].cpu()-DBalanced[k].cpu()))/abs(DUnbalanced[k].cpu())).flatten())
    
DConvDelta={}    
for i in range(0,5):
    r=re.compile("blocks\.{}\.0\.conv.*".format(i))
    query=list(filter(r.match,DDelta))
    convSum=0
    count=0
    for j in query:
        if j.find("sc")==-1: #Remove short cuts
            print("Removed short cut")
            convSum+=DDelta[j]
            count+=1
    #Find the average in the block
    convSum=convSum/count
    DConvDelta["block{}.conv".format(i)]=convSum*100

#General plots
#Generator
plt.barh(list(GDelta.keys()),[i.numpy()*100 for i in list(GDelta.values())])
plt.title("Generator")
plt.xlabel("Perecentage Change, %")
plt.show()

#Discriminator
plt.barh(list(DDelta.keys()),[i.numpy()*100 for i in list(DDelta.values())])
plt.title("Discriminator")
plt.xlabel("Perecentage Change, %")
plt.show()

tickSize=24
fig=plt.figure(figsize=(20,15))

#Summary plots (Generator)
ax1=plt.subplot(211)
ax1.bar(GConvDelta.keys(),GConvDelta.values())
plt.xticks(fontsize=tickSize)
plt.yticks(fontsize=tickSize)
ax1.set_ylabel("Perecentage Change, %",fontsize=tickSize)
ax1.set_title("Generator",fontsize=tickSize)


#Summary plots (Discriminator)
ax2=plt.subplot(212)
ax2.bar(DConvDelta.keys(),DConvDelta.values())
plt.xticks(fontsize=tickSize)
plt.yticks(fontsize=tickSize)
ax2.set_ylabel("Perecentage Change, %",fontsize=tickSize)
ax2.set_title("Discriminator",fontsize=tickSize)

# fig

# plt.bar(convDelta.keys(),convDelta.values())

#Load Model
# G = model.Generator(**config)
# GStateDict=G.state_dict()