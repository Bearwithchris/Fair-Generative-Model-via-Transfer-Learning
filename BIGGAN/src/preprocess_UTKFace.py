import torch
import tqdm
import cv2
import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# total number of examples: 23705
# white: 10078
# non-white: 13627
# white 10078, black 4526, asian 3434, indian 3975, others 1692

IMG_SIZE = 64

dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())

split_ratio = {'train': 0.8, 'valid': 0.2, 'test': 0}
# 8 : 1 : 1 for training attr. clf
# 8 : 2 for training DR clf

def preprocess_images(args):
    # automatically save outputs to "data" directory
    IMG_PATH = 'UTK_{0}x{0}.pt'.format(IMG_SIZE)
    LABEL_PATH = 'labels_UTK_{0}x{0}.pt'.format(IMG_SIZE)
    
    # NOTE: datasets have not yet been normalized to lie in [-1, +1]!
    transform = transforms.Resize(IMG_SIZE)

    df = parse_dataset(args.data_dir)
    file_list = list(df['file'])
    race_list = list(df['race'])

    N_IMAGES = len(file_list)
    data = np.zeros((N_IMAGES, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
    labels = np.zeros((N_IMAGES))

    print('starting conversion...')
    for i in tqdm.tqdm(range(N_IMAGES)):
        with Image.open(file_list[i]) as img:
            if transform is not None:
                img = transform(img)
        img = np.array(img)
        data[i] = img.transpose((2, 0, 1))
        
        if race_list[i] is 'white':
            labels[i] = 0
            
        elif race_list[i] is 'black':
            labels[i] = 1
            
        elif race_list[i] is 'asian':
            labels[i] = 2
            
        elif race_list[i] is 'indian':
            labels[i] = 3
            
        elif race_list[i] is 'others':
            labels[i] = 4

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    train_data, train_labels,\
    valid_data, valid_labels,\
    test_data, test_labels = UTK_split(data, labels, split_ratio)
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
                
    print("Saving full-set images to full_{}".format(IMG_PATH))
    torch.save(data, os.path.join(args.out_dir,'full_{}'.format(IMG_PATH)))
    torch.save((labels >= 1).double(), os.path.join(args.out_dir,'full_{}'.format(LABEL_PATH)))

    print("Saving train images to train_{}".format(IMG_PATH))
    torch.save(train_data, os.path.join(args.out_dir,'train_{}'.format(IMG_PATH)))
    torch.save(train_labels, os.path.join(args.out_dir,'train_{}'.format(LABEL_PATH)))

    print("Saving valid images to val_{}".format(IMG_PATH))
    torch.save(valid_data, os.path.join(args.out_dir,'val_{}'.format(IMG_PATH)))
    torch.save(valid_labels, os.path.join(args.out_dir,'val_{}'.format(LABEL_PATH)))

    print("Saving test images to test_{}".format(IMG_PATH))
    torch.save(test_data, os.path.join(args.out_dir,'test_{}'.format(IMG_PATH)))
    torch.save(test_labels, os.path.join(args.out_dir,'test_{}'.format(LABEL_PATH)))

def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    
    return df


def UTK_split(data, labels, split_ratio):
    n_white = (labels==0).sum().item()
    n_black = (labels==1).sum().item()
    n_asian = (labels==2).sum().item()
    n_indian = (labels==3).sum().item()
    n_others = (labels==4).sum().item()

    white_count = 0
    black_count = 0
    asian_count = 0
    indian_count = 0
    others_count = 0

    train_split = split_ratio['train']
    valid_split = split_ratio['valid']
    test_split = split_ratio['test']

    train_data = []
    train_labels = []

    valid_data = []
    valid_labels = []

    test_data = []
    test_labels = []

    for i in range(len(labels)):
        if labels[i] == 0:
            if white_count < n_white * train_split:
                train_data.append(data[i])
                train_labels.append(labels[i])
                white_count += 1

            elif (white_count >= n_white * train_split) and (white_count < n_white * (train_split + valid_split)):
                valid_data.append(data[i])
                valid_labels.append(labels[i])
                white_count += 1

            else:
                test_data.append(data[i])
                test_labels.append(labels[i])
                white_count += 1

        elif labels[i] == 1:
            if black_count < n_black * train_split:
                train_data.append(data[i])
                train_labels.append(labels[i])
                black_count += 1

            elif (black_count >= n_black * train_split) and (black_count < n_black * (train_split + valid_split)):
                valid_data.append(data[i])
                valid_labels.append(labels[i])
                black_count += 1

            else:
                test_data.append(data[i])
                test_labels.append(labels[i])
                black_count += 1

        elif labels[i] == 2:
            if asian_count < n_asian * train_split:
                train_data.append(data[i])
                train_labels.append(1) # we consider binary sensitive attributes
                asian_count += 1

            elif (asian_count >= n_asian * train_split) and (asian_count < n_asian * (train_split + valid_split)):
                valid_data.append(data[i])
                valid_labels.append(1)
                asian_count += 1

            else:
                test_data.append(data[i])
                test_labels.append(1)
                asian_count += 1

        elif labels[i] == 3:
            if indian_count < n_indian * train_split:
                train_data.append(data[i])
                train_labels.append(1) # we consider binary sensitive attributes
                indian_count += 1

            elif (indian_count >= n_indian * train_split) and (indian_count < n_indian * (train_split + valid_split)):
                valid_data.append(data[i])
                valid_labels.append(1)
                indian_count += 1

            else:
                test_data.append(data[i])
                test_labels.append(1)
                indian_count += 1

        else:
            if others_count < n_others * train_split:
                train_data.append(data[i])
                train_labels.append(1) # we consider binary sensitive attributes
                others_count += 1

            elif (others_count >= n_others * train_split) and (others_count < n_others * (train_split + valid_split)):
                valid_data.append(data[i])
                valid_labels.append(1)
                others_count += 1

            else:
                test_data.append(data[i])
                test_labels.append(1)
                others_count += 1

    train_data_tensor = torch.zeros((len(train_data), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    train_labels_tensor = torch.zeros((len(train_data)))
    for i in range(len(train_data)):
        train_data_tensor[i] = train_data[i]
        train_labels_tensor[i] = train_labels[i]

    valid_data_tensor = torch.zeros((len(valid_data), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    valid_labels_tensor = torch.zeros((len(valid_data)))
    for i in range(len(valid_data)):
        valid_data_tensor[i] = valid_data[i]
        valid_labels_tensor[i] = valid_labels[i]

    test_data_tensor = torch.zeros((len(test_data), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    test_labels_tensor = torch.zeros((len(test_data)))
    for i in range(len(test_data)):
        test_data_tensor[i] = test_data[i]
        test_labels_tensor[i] = test_labels[i]
        
    return train_data_tensor, train_labels_tensor, \
            valid_data_tensor, valid_labels_tensor, \
            test_data_tensor, test_labels_tensor
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/UTKFace', type=str, 
        help='path to downloaded UTKFace dataset (e.g. /data/UTKFace/')
    parser.add_argument('--out_dir', default='../data/UTKFace', type=str, 
        help='destination of outputs')
    args = parser.parse_args()
    preprocess_images(args)
