# -*- coding: utf-8 -*-

import pickle
import shutil
import os
import numpy as np
import PIL
from tqdm import tqdm
import torch
from torchvision import transforms

resize = transforms.Compose([transforms.Scale((64,64))])


attributeList=["Gender","Blackhair","Young","Moustache","Smiling"]
# i=1
totalSamples=70000
perc=0.02

refSamplesCountDict={0.04:totalSamples*0.04,0.02:totalSamples*0.02}
refSamplesCount=refSamplesCountDict[perc]

#Generate paths
for files in attributeList:
    
    if not os.path.isdir("./training"):
        os.mkdir("./training")
    if not os.path.isdir("./training/%s"%files):
        os.mkdir("./training/%s"%files)
        os.mkdir("./training/%s/0"%files)
        os.mkdir("./training/%s/1"%files)
        os.mkdir("./training/%s/%sFIDSamples"%(files,files))
    if not os.path.isdir("./training/%s/%sTrainSamples_%s"%(files,files,str(perc))):
        os.mkdir("./training/%s/%sTrainSamples_%s"%(files,files,str(perc)))
    # except:
    #     print ("%s Path exists"%files)


#Load and distribute data
for i in range(len(attributeList)):

    with open('../ffhq-features-dataset-master/FFHQ%sLabels.pkl'%attributeList[i], 'rb') as f:
        loaded_dict = pickle.load(f)
        
    labels=np.array(list(loaded_dict.values()))
    label0=np.where(labels==0)[0]
    label1=np.where(labels==1)[0]
    FIDLabelCount=min(len(label0),len(label1))
    
    
    if FIDLabelCount>=refSamplesCount/2:
        label0Keys=np.array(list(loaded_dict.keys()))[label0]
        label1Keys=np.array(list(loaded_dict.keys()))[label1]
        
        #Form FID Balanced List to comapre against
        FIDLabelList=np.concatenate((label0Keys[0:FIDLabelCount],label1Keys[0:FIDLabelCount]))
        
        #Form Ref samples list
        refSample=np.concatenate((label0Keys[0:int(refSamplesCount/2)],label1Keys[0:int(refSamplesCount/2)]))
        
        clf_training_data=[]
        clf_training_labels=[]
        #Debug purposes to visualise the datas split
        print ("Generating Samples for %s"%attributeList[i])
        for key,values in tqdm(loaded_dict.items()):
            # shutil.copy("./data/resized/"+key,"./training/%s/%s/%s"%(attributeList[i],values,key))
            if key in FIDLabelList:
                # shutil.copy("./data/resized/"+key,"./training/%s/%sFIDSamples/%s"%(attributeList[i],attributeList[i],key))
                clf_training_data.append(np.array(resize((PIL.Image.open("./data/resized/"+key)))))
                clf_training_labels.append(values)              
            # if key in refSample:
            #     shutil.copy("./data/resized/"+key,"./training/%s/%sTrainSamples_%s/%s"%(attributeList[i],attributeList[i],str(perc),key))
                
        #Save clf training data
        clf_training_data=torch.tensor(clf_training_data).permute(0,3,1,2)
        clf_training_labels=torch.tensor(clf_training_labels)
        torch.save(clf_training_data,"%s_FFHQ_Clf_Training_data.pt"%attributeList[i])
        torch.save(clf_training_labels,"%s_FFHQ_Clf_Training_labels.pt"%attributeList[i])
        
        
    else:
        print ("Not enough samples, Min balanced per attribute= %i"%FIDLabelCount)
        
        