''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import time
import os
import glob
import sys

import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import argparse

import metrics as fd
from torchvision import transforms,models
# Import my stuff
# import inception_utils
# import utils
# import utils_add_on as uao
# import losses
# from clf_models import ResNet18, BasicBlock, Net

import sys
# sys.path.append('../Data_prep_Train')
# from clf_models import ResNet18, BasicBlock, Net


def mobilenet():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

# def fairness_discrepancy(data, n_classes):
#     """
#     computes fairness discrepancy metric for single or multi-attribute
#     this metric computes L2, L1, AND KL-total variation distance
#     """
#     unique, freq = np.unique(data, return_counts=True)
#     props = freq / len(data) #Proportion of data that belongs to that data
#     print (freq)
#     truth = 1./n_classes


#     # L2 and L1
#     l2_fair_d = np.sqrt(((props - truth)**2).sum())/n_classes
#     l1_fair_d = abs(props - truth).sum()/n_classes

#     # q = props, p = truth
#     kl_fair_d = (props * (np.log(props) - np.log(truth))).sum()

#     #Cross entropy
#     p=np.ones(n_classes)/n_classes    
#     # ce=cross_entropy(p,props,n_classes)-cross_entropy(p,p,n_classes)
    
#     #information specificity
#     rank=np.linspace(1,n_classes-1,n_classes-1)
#     rank[::-1].sort() #Descending order
#     perc=np.array([i/np.sum(rank) for i in rank])
    
#     #Create array to populate proportions
#     props2=np.zeros(n_classes)
#     props2[unique]=props
                  
#     props2[::-1].sort()
#     alpha=props2[1:]
#     specificity=abs(props2[0]-np.sum(alpha*perc))
#     info_spec=(l1_fair_d+specificity)/2
    
    
#     return l2_fair_d, l1_fair_d, kl_fair_d,info_spec,specificity


def load_data(attributes,index):
    data=torch.load('../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    print ("Data loaded: "+'../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    dataset=data[0]
    dataset=preprocess(dataset)
    labels=data[1]
    train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(data[0]),data[1])

def classify_examples(model, sample_path,X,load_save):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    probs = []
    if load_save==1:
        samples = np.load(sample_path)['x']
    else:
        samples=X
    bs=10
    n_batches = samples.shape[0] // bs
    remainder=samples.shape[0]-(n_batches*bs)
    print (sample_path)

    with torch.no_grad():
        # generate 10K samples
    
        for i in range(n_batches):
            x = samples[i*bs:(i+1)*bs]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits= model(samp)
            probas=F.softmax(logits)
            
            
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
          
            
        if remainder!=0:
            x = samples[(i+1)*bs:(bs*(i+1))+remainder]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()
            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
            
            
            
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()
        # probs = torch.cat(probs).data.cpu()

    return preds, probs

def run():
    # Prepare state dict, which holds things like epoch # and itr #
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_path', type=str, help='Folder of the CLF file', default="attr_clf")
    parser.add_argument('--multi_clf_path', type=str, help='Folder of the Multi CLF file', default="multi_clf_2")
    parser.add_argument('--index', type=int, help='dataset index to load', default=0)
    parser.add_argument('--class_idx', type=int, help='CelebA class label for training.', default=20)
    # parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8,20])
    parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[20])
    parser.add_argument('--multi', type=int, default=1, help='If True, runs multi-attribute classifier')
    parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
    parser.add_argument('--load_save', type=int, help='To generate the individial dataset npz', default=0)
    args = parser.parse_args()


    CLF_PATH = './results/%s/model_best.pth.tar'%args.clf_path
    MULTI_CLF_PATH = './results/%s/model_best.pth.tar'%args.multi_clf_path
    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    #Transform image
    

    #Log Runs
    f=open('../%s/log_stamford_fair.txt' %("logs"),"a")
    fnorm=open('../%s/log_stamford_fair_norm.txt' %("logs"),"a")
    data_log=open('../%s/log_stamford_fair_norm_raw.txt' %("logs"),"a")

    # experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
    #                     else utils.name_from_config(config))
    
    #Load dataset to be tested
    if args.multi==1:
        attributes=2**len(args.multi_class_idx)
    else: 
        attributes=2 #Single class
    train_set,size,labels=load_data(attributes, args.index)
    
    # # classify examples and get probabilties
    # n_classes = 2
    # if config['multi']:
    #     n_classes = 4
   
    print ("Preparing data....")
    print ("Dataset has a total of %i data instances"%size)
    k=0
    
    file="../data/FID_sample_storage_%i"%attributes
    if (os.path.exists(file)!=True):
        os.makedirs(file)
    npz_filename = '%s/%s_fid_real_samples_%s.npz' % (file,attributes, args.index) #E.g. perc_fid_samples_0
    if os.path.exists(npz_filename):
        print('samples already exist, skipping...')
    else:
        X = []
        pbar = tqdm(train_set)
        print('Sampling images and saving them to npz...' ) #10k
        count=1 
        
        for i ,x in enumerate(pbar):
            X+=x                
    
        X=np.array(torch.stack(X)).astype(np.uint8)
        if args.load_save==1:
            print('Saving npz to %s...' % npz_filename)
            np.savez(npz_filename, **{'x': X})
                
   

    #=====Classify===================================================================
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)

    # output file
    # fname = '%s/%s_fair_disc_fid_samples.p' % (config['samples_root'], perc_bias)

    # load classifier 
    #(Saved state)
    if args.multi==0:
        print('Pre-loading pre-trained single-attribute classifier...')
        clf_state_dict = torch.load(CLF_PATH)['state_dict']
        clf_classes = attributes
    else:
        # multi-attribute
        print('Pre-loading pre-trained multi-attribute classifier...')
        clf_state_dict = torch.load(MULTI_CLF_PATH)['state_dict']
        clf_classes = attributes
        
    # load attribute classifier here
    #(Model itself)
    # clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=attributes, grayscale=False) 
    clf=mobilenet()
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # classify examples and get probabilties
    # n_classes = 2
    # if config['multi']:
    #     n_classes = 4

    # number of classes
    probs_db = np.zeros((1, size, clf_classes)) #Numper of runs , images per run ,Number of classes
    for i in range(1):
        # grab appropriate samples
        if args.load_save==1:
            npz_filename = os.path.join("../data","FID_sample_storage_%i"%attributes,'%s_fid_real_samples_%s.npz' % (attributes, args.index))
            preds, probs = classify_examples(clf, npz_filename,X,args.load_save) #Classify the data
        else:
            preds, probs = classify_examples(clf, npz_filename,X,args.load_save) #Classify the data
        
        
        #Log the prediction for error analysis
        dist=fd.pred_2_dist(preds,clf_classes)
        if (os.path.isfile("../logs/pred_dist.npz")):
            x=np.load("../logs/pred_dist.npz")["x"]
            x=np.vstack((x,dist))
            np.savez("../logs/pred_dist.npz",x=x)
        else:
            #if there doesn't already exist this file
            np.savez("../logs/pred_dist.npz",x=dist)
        
        
        # l2, l1, kl,IS,specificity = fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        l2, l1,IS,specificity,wd,wds= fd.fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        # l2_norm, l1_norm,IS_norm,specificity_norm, wd_norm,wds_norm= fd.fairness_discrepancy(preds, clf_classes,1) #Pass to calculate score
        
        #exp
        # l2Exp, l1Exp, klExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # save metrics (To add on new mertrics add here)
        l2_db[i] = l2
        l1_db[i] = l1
        # kl_db[i] = kl
        probs_db[i] = probs
        
        #Write log
        # f.write("Running: "+npz_filename+"\n")
        f.write('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={}, wd={}, wds={} \n'.format(attributes,args.index, l2, l1, IS,specificity,wd,wds))  
        # fnorm.write('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={}, wd={} \n'.format(attributes,args.index, l2_norm, l1_norm, IS_norm,specificity_norm,wd_norm))
        # print('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={}, wd={} \n'.format(attributes,args.index, l2, l1, IS,specificity,wd))
        # # print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2Exp, l1Exp, klExp))
        
        # f.write('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={} \n'.format(attributes,args.index, l2, l1, IS,specificity))  
        # fnorm.write('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={}, , wd={}, wds={} \n'.format(attributes,args.index, l2_norm, l1_norm, IS_norm,specificity_norm,wd_norm,wds_norm))
        # data_log.write("{},{},{},{},{},{},{},{}\n".format(attributes,args.index, l2_norm, l1_norm, IS_norm,specificity_norm,wd_norm,wds_norm))
        # print('Fair_disc for classes {} index {} is: l2={} l1={} IS={} Specificity={}, , wd={}, wds={} \n'.format(attributes,args.index, l2_norm, l1_norm, IS_norm,specificity_norm,wd_norm,wds_norm))
        
        
        #Commented out for stamford experiment*************************************************************************************************
        # #FID score 50_50 vs others 
        # data_moments=os.path.join("./samples","0.5_fid_real_samples_ref_0.npz")
        # sample_moments=os.path.join("./samples",'%s_fid_real_samples_%s.npz'%(perc_bias,k))
        # # FID = fid_score_mod.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
        # FID = fid_score_mod_AE.calculate_faed_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)

        # print ("FID: "+str(FID))
        # f.write("FID: "+str(FID)+"\n")     
        #Commented out for stamford experiment*************************************************************************************************
        
        f.close()
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    # print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    print(l1_db)
    print(IS)
    print(specificity)
    print (wd)
    
    # # save all metrics
    # with open(fname, 'wb') as fp:
    #     pickle.dump(metrics, fp)
    # np.save(os.path.join(config['samples_root'], 'clf_probs.npy'), probs_db)


# def main():
#     # parse command line and run
#     parser = utils.prepare_parser()
#     parser = utils.add_sample_parser(parser)
#     config = vars(parser.parse_args())
#     print(config)
#     run(config)


if __name__ == '__main__':
    preprocess = transforms.Compose([transforms.Resize(224)])
    run()
#     main()
