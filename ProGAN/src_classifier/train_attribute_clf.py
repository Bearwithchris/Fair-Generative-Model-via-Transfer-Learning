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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='celeba',default='celeba',)
    parser.add_argument('--out_dir', type=str, help='where to save outputs',default="./results/attr_clf")
    parser.add_argument('--ckpt-path', type=str, default='./results/multi_clf', 
                        help='if test=True, path to clf checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='minibatch size [default: 64]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [default: 10]')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='number of steps to log after during training')
    # parser.add_argument('--class_idx', type=int, default=8,
    #                     help='CelebA class label for training.')
    parser.add_argument('--class_name', type=str, default="Male",
                        help='CelebA class label for training.')
    parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8,20])
    parser.add_argument('--multi', type=bool, default=False, 
						help='If True, runs multi-attribute classifier')
    parser.add_argument('--even', type=int, default=1, 
                        help='If True, runs multi-even-attribute classifier')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--count', type=int, default=10000,
                        help='Count of training data')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # reproducibility
    torch.manual_seed(777)
    np.random.seed(777)

    device = torch.device('cuda' if args.cuda else 'cpu')
    
    #Transform image
    preprocess = transforms.Compose([transforms.Resize(224)])
    
    out_dir=os.path.join(args.out_dir,str(args.class_name))
      #Create output folder
    # if not os.path.isdir(args.out_dir):
    os.makedirs(out_dir,exist_ok=True)

    # get data: idx 20 = male, idx 8 = black hair
    if not args.multi:
	    if args.even==True:
    		train_dataset = build_even_celeba_classification_dataset_name(
    			'train', args.class_name)
    		valid_dataset = build_even_celeba_classification_dataset_name(
    			'val', args.class_name)
    		test_dataset = build_even_celeba_classification_dataset_name(
    			'test', args.class_name)
	    else:
    		train_dataset = build_celeba_classification_dataset(
    			'train', args.class_idx)
    		valid_dataset = build_celeba_classification_dataset(
    			'val', args.class_idx)
    		test_dataset = build_celeba_classification_dataset(
    			'test', args.class_idx)
	    n_classes = 2
    else:
        #(Dataset are already in torch.utils.data format)
        train_dataset = build_multi_celeba_classification_datset('train')
        valid_dataset = build_multi_celeba_classification_datset('val')
        test_dataset = build_multi_celeba_classification_datset('test')
        n_classes = 4


    f=open("./logs/Attribute_classifier_log.txt","a")
    f.write(str(len(train_dataset))+" "+str(len(valid_dataset)))
    print(len(train_dataset))
    print(len(valid_dataset))

    # train/validation split (Shuffle and batch the datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # build model
    # model_cls = build_model(args.model_name)
    # model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
#     model=Net(n_classes)
    model = resnet18()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #Pass model to cuda
    # model.to(device)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #Pass data to cuda
            data, target = preprocess(data).to(device), target.to(device)
            #Normalise data
            data = data.float() / 255.
            target = target.long()
            
            # NOTE: here, female (y=0) and male (y=1)
            # print (model(data))
            logits= model(data)
            probas=F.softmax(logits)
            loss = F.cross_entropy(logits, target)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             if batch_idx % args.log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader), loss.item()))


    def test(epoch, loader):
        model.eval()
        test_loss = 0
        correct = 0
        num_examples = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = preprocess(data).to(device), target.to(device)
                data = data.float() / 255.
                target = target.long()

                # run through model
                # logits, probas = model(data)
                logits= model(data)
                probas=F.softmax(logits)
                # print (logits[0])
                # print (target[0])
                test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
                _, pred = torch.max(probas, 1)
                num_examples += target.size(0)
                correct += (pred == target).sum()

        test_loss /= num_examples
        f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, num_examples,100. * correct / num_examples))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, num_examples,
            100. * correct / num_examples))
        return test_loss

    # classifier has finished training, evaluate sample diversity
    best_loss = sys.maxsize

    #Training Loop
    print('beginning training...')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid_loss = test(epoch, valid_loader)

        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        state_dict = model.state_dict()
        if is_best:
            print('saving checkpoint at epoch {}'.format(epoch))
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'cmd_line_args': args,
            }, is_best, epoch, folder=out_dir)
            best_idx = epoch
            best_state = model.state_dict()

    # finished training, want to test on final test set
    print('finished training...testing on final test set with epoch {} ckpt'.format(best_idx))
    
    # reload best model
    # model_cls = build_model('celeba')
    # model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
    # model = model.to(device)
    model = resnet18()
     # model=Net(n_classes)
    model.cuda()
    model.load_state_dict(best_state)

    # get test
    test_loss = test(epoch, test_loader)