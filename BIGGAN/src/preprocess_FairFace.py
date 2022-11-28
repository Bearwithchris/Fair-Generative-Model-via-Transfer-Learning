import torch
import tqdm
import numpy as np 
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

IMG_SIZE = 64

def preprocess_images(args):
    train_img_path = '{0}_FairFace_{1}x{1}.pt'.format('train', IMG_SIZE)
    train_label_path = '{0}_labels_FairFace_{1}x{1}.pt'.format('train', IMG_SIZE)

    train_path_data, train_attr_data = load_path_attr('train', args.data_dir)

    transform = transforms.Resize(IMG_SIZE)

    train_n_images = 28760 # white 16527 + black 12233 
    train_data = np.zeros((train_n_images, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
    train_labels = np.zeros((train_n_images))
    
    train_count = 0

    print('starting conversion of training data...')
    for i in tqdm.tqdm(range(len(train_path_data))):
        with Image.open(os.path.join(args.data_dir, '{}'.format(train_path_data[i]))) as img:
            if transform is not None:
                img = transform(img)
        img = np.array(img)
        if train_attr_data[i] == 'White':
            train_labels[train_count] = 0 # label 0 for white examples
            train_data[train_count] = img.transpose((2, 0, 1))
            train_count += 1
        elif train_attr_data[i] == 'Black':
            train_labels[train_count] = 1 # label 1 for black examples
            train_data[train_count] = img.transpose((2, 0, 1))
            train_count += 1

    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    print("Saving training images to {}".format(train_img_path))
    torch.save(train_data, os.path.join(args.out_dir, train_img_path))
    torch.save(train_labels, os.path.join(args.out_dir, train_label_path))
    
    if args.split_test:
        val_img_path = '{0}_FairFace_{1}x{1}.pt'.format('val', IMG_SIZE)
        val_label_path = '{0}_labels_FairFace_{1}x{1}.pt'.format('val', IMG_SIZE)

        test_img_path = '{0}_FairFace_{1}x{1}.pt'.format('test', IMG_SIZE)
        test_label_path = '{0}_labels_FairFace_{1}x{1}.pt'.format('test', IMG_SIZE)

        val_path_data, val_attr_data, test_path_data, test_attr_data \
        = load_path_attr('val', args.data_dir, split_test=args.split_test)

        transform = transforms.Resize(IMG_SIZE)

        val_n_images = 2175 #  white 1351 + black 824
        val_data = np.zeros((val_n_images, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
        val_labels = np.zeros((val_n_images))
        
        val_count = 0

        test_n_images = 1466 # white 734 + black 732
        test_data = np.zeros((test_n_images, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
        test_labels = np.zeros((test_n_images))
        
        test_count = 0

        print('starting conversion of validation data...')
        for i in tqdm.tqdm(range(len(val_path_data))):
            with Image.open(os.path.join(args.data_dir, '{}'.format(val_path_data[i]))) as img:
                if transform is not None:
                    img = transform(img)
            img = np.array(img)
            if val_attr_data[i] == 'White':
                val_labels[val_count] = 0
                val_data[val_count] = img.transpose((2, 0, 1))
                val_count += 1
            elif val_attr_data[i] == 'Black':
                val_labels[val_count] = 1
                val_data[val_count] = img.transpose((2, 0, 1))
                val_count += 1

        print('starting conversion of test data...')
        for i in tqdm.tqdm(range(len(test_path_data))):
            with Image.open(os.path.join(args.data_dir, '{}'.format(test_path_data[i]))) as img:
                if transform is not None:
                    img = transform(img)
            img = np.array(img)
            if test_attr_data[i] == 'White':
                test_labels[test_count] = 0
                test_data[test_count] = img.transpose((2, 0, 1))
                test_count += 1
            elif test_attr_data[i] == 'Black':
                test_labels[test_count] = 1
                test_data[test_count] = img.transpose((2, 0, 1))
                test_count += 1

        val_data = torch.from_numpy(val_data)
        val_labels = torch.from_numpy(val_labels)

        test_data = torch.from_numpy(test_data)
        test_labels = torch.from_numpy(test_labels)

        print("Saving validation images to {}".format(val_img_path))
        torch.save(val_data, os.path.join(args.out_dir, val_img_path))
        torch.save(val_labels, os.path.join(args.out_dir, val_label_path))

        print("Saving test images to {}".format(test_img_path))
        torch.save(test_data, os.path.join(args.out_dir, test_img_path))
        torch.save(test_labels, os.path.join(args.out_dir, test_label_path))

    else:
        val_img_path = '{0}_FairFace_{1}x{1}.pt'.format('val', IMG_SIZE)
        val_label_path = '{0}_labels_FairFace_{1}x{1}.pt'.format('val', IMG_SIZE)

        val_path_data, val_attr_data = load_path_attr('val', args.data_dir, split_test=args.split_test)

        transform = transforms.Resize(IMG_SIZE)

        val_n_images = 3641 # white 2085 + black 1556
        val_data = np.zeros((val_n_images, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
        val_labels = np.zeros((val_n_images))
        
        val_count = 0

        print('starting conversion of validation data...')
        for i in tqdm.tqdm(range(len(val_path_data))):
            with Image.open(os.path.join(args.data_dir, '{}'.format(val_path_data[i]))) as img:
                if transform is not None:
                    img = transform(img)
            img = np.array(img)
            if val_attr_data[i] == 'White':
                val_labels[val_count] = 0
                val_data[val_count] = img.transpose((2, 0, 1))
                val_count += 1
            elif val_attr_data[i] == 'Black':
                val_labels[val_count] = 1
                val_data[val_count] = img.transpose((2, 0, 1))
                val_count += 1

        val_data = torch.from_numpy(val_data)
        val_labels = torch.from_numpy(val_labels)

        print("Saving validation images to {}".format(val_img_path))
        torch.save(val_data, os.path.join(args.out_dir, val_img_path))
        torch.save(val_labels, os.path.join(args.out_dir, val_label_path))

        
def load_path_attr(partition, data_dir, split_test=0):
    path_data = []
    attr_data = []
    if split_test:
        path_data_test = []
        attr_data_test = []
    
    with open(os.path.join(data_dir, 'fairface_label_{}.csv'.format(partition))) as fp:
        rows = fp.readlines()
        for row in rows[1:]:
            path, _, _, race, test = row.strip().split(',')
            if not split_test:
                path_data.append(path)
                attr_data.append(race)
            else:
                if test == 'False':
                    path_data.append(path)
                    attr_data.append(race)
                else:
                    path_data_test.append(path)
                    attr_data_test.append(race)
                    
    if not split_test:
        return path_data, attr_data
    else:
        return path_data, attr_data, path_data_test, attr_data_test
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/FairFace', type=str, 
        help='path to downloaded FairFace dataset (e.g. /data/FairFace/')
    parser.add_argument('--out_dir', default='../data/FairFace', type=str, 
        help='destination of outputs')
    parser.add_argument('--split_test', default=0, type=int, 
        help='whether to generate test split by halving val set')
    args = parser.parse_args()
    preprocess_images(args)