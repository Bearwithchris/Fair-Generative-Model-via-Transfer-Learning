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

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    """
    fixed to take in density ratio estimates
    """
    # properly match up dimensions, and only reweight real examples
    # loss_real = torch.mean(F.relu(1. - dis_real))
    
    #Attempt 1
    weighted = F.relu(1. - dis_real)
    loss_real = torch.mean(weighted)
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    
    #Doesn't work
    # weighted = (1. - dis_real)
    # loss_real = torch.mean(weighted)
    # loss_fake = torch.mean(1. + dis_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    # with torch.no_grad():
    #   dis_fake_norm = torch.exp(dis_fake).mean()
    #   dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    # dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss

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
        PATH="../output/state_dict_%i.pth"%loadFromIndex
        model.load_state_dict(torch.load(PATH))
    
    return model

def train(model,GOptim,DOptim,trainloader,batchSize,numDperGUpdates):
    #set all models and data to cude
    device = 'cuda'
    # G=G.to(device)
    # D=D.to(device)
    # dataX.to(device)
    # dataY.to(device)
    
    #Intiialize the optimizer
    GOptim.zero_grad()
    DOptim.zero_grad()
    # x = torch.split(dataX, batchSize)
    # y = torch.split(dataY, batchSize)
    
    for i, data in enumerate(trainloader):    
        for j in range(numDperGUpdates):
            DOptim.zero_grad()
            #Discriminate real Data
            Dreal=model.netD(data[0].to(device))
            
            #Discriminate fake Data
            noise, _ = model.buildNoiseData(batchSize)
            fake_images=model.netG(noise.to(device))
            Dfake=model.netD(fake_images)
            
            #Update discriminator
            D_loss_real, D_loss_fake = loss_hinge_dis(Dfake,Dreal)
            D_loss = (D_loss_real + D_loss_fake)     
            D_loss.backward()
            optimD.step()
            
        
        
        #Update Generator
        GOptim.zero_grad()
        fake_images=model.netG(noise.to(device))
        Dfake=model.netD(fake_images)
        G_loss = loss_hinge_gen(Dfake)    
        G_loss.backward()
        optimG.step()
        
        
        
        
#Hyper Parameters      
lrG=2e-4
lrD=1e-4
batch_size=8
epochs=60
numDperGUpdates=2

#0) Load Prepare D-ref dataset
path="../../Data"
# x=torch.load(os.path.join(path,"CelebAHQ_0.025000_Male_data.pt"))
x=torch.load(os.path.join(path,"CelebAHQ_0.050000_Black_Hair_data.pt"))
# y=torch.load(os.path.join(path,"CelebAHQ_0.025000_Male_labels.pt"))

#Pre-process x into [-1,1]
transformList = [#NumpyResize(size),
                 #NumpyToTensor(),
                 Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms = Transforms.Compose(transformList)
x=(x/255.0) #Normalise to [0,1] as per dataset.ImageFolder
x=transforms(x.float())

#Comments: Need to edit (now the range isn't between 0-1)
# x=transform(x.float())
trainData=torch.utils.data.TensorDataset(torch.tensor(x))
# trainData.dataset.transform=Transforms.Compose([ Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainloader = DataLoader(trainData, batch_size=batch_size,
                              shuffle=True)
#1) Load pre-trained model
loadFromIndex=None
# Naive Transfer loading
model=PGAN(pretrained=True, model_name='celebAHQ-256', loadFrom=loadFromIndex )
    
# G=model.getNetG()
# D=model.getNetD()

#Optimizer
optimG = optim.Adam(model.netG.parameters(),lr=lrG)
optimD= optim.Adam(model.netD.parameters(),lr=lrD)
if loadFromIndex!=None:
    optimG.load_state_dict("../output/optimG_dict_%i.pth"%loadFromIndex)
    optimD.load_state_dict("../output/optimD_dict_%i.pth"%loadFromIndex)
#2) Evaluate debiased transfer-learning Model 

#LOad Fixed Noise
num_images = 56
noisePath="../output/savedNoise%i.pth"%num_images
if not os.path.isfile(noisePath):
    noise, _ = model.buildNoiseData(num_images)
    torch.save(noise,noisePath)
else:
    noise=torch.load(noisePath)
    print ("Save Z Loaded")

#Load old checkpoint
if loadFromIndex==None:
    start=0
else:
    start=loadFromIndex
    
for i in tqdm(range(start,epochs+1)):
    # Plot example
    with torch.no_grad():
        generated_images = model.test(noise)
    
    # let's plot these images using torchvision and matplotlib
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.imsave("../output/generatedImages_%i.jpg"%i , grid.permute(1, 2, 0).cpu().numpy())
    # plt.show()
    
    # #save Model
    if i%10==0:
        model.save("../output/state_dict_%i.pth"%i)
        torch.save(optimG.state_dict(), "../output/optimG_dict_%i.pth"%i)
        torch.save(optimD.state_dict(), "../output/optimD_dict_%i.pth"%i)
    
    
    del(generated_images)
    del(grid)
    
    train(model,optimG,optimD,trainloader,batch_size,numDperGUpdates)