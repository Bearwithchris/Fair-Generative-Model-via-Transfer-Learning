# a) Fair Generative Model Via Transfer Learning
> Model Base codes were adopted from Choi et.al https://github.com/ermongroup/fairgen
> Preprocessing of UTKFace and some data pre-processing steps were adopted from Um et.al https://openreview.net/pdf?id=F1Z3QH-VjZE
> Use Files in "./BIGGAN"

#Training FairGAN for Validation
(Read me is abstracted from Choi etal source code and some modifications have been made for our purpose)
## 1) Data setup:
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.
(b) Download the UTKFace dataset here (https://susanqq.github.io/UTKFace/) into the `data/` directory
(c) Preprocess the CelebA dataset for faster training:
```
python preprocess_celeba.py --data_dir=/path/to/downloaded/dataset/celeba/ --out_dir=../data --partition=train
```
You should run this script for `--partition=[train, val, test]` to cache all the necessary data. The preprocessed files will then be saved in `data/`.
(d) Preprocess the UTKFace dataset for fast training
```
python3 preprocess_UTKFace.py --data_dir=/path/to/downloaded/dataset/UTKFace/ --out_dir=../data/UTKFace
```
e)FID: For CelebA, we have provided unbiased FID statistics in the source directory. For Multi CelebA run ./BIGGAN/notebook/multi-attribute data and unbiased FID splits.ipynb . For UTKFace run ./BIGGAN/notebook/UTKFace unbiased FID splits.ipynb
f)


#Preprocessing files for the given LA & Training classifier
## 2) For training of Resnet Classifier
a) Train attribute classifier with CelebA (Single Attribute)
```
python train_attribute_clf.py celeba ./results/celeba/attr_clf
```
a) Train attribute classifier with CelebA (Multi Attribute)
```
python train_attribute_clf.py celeba ./results/celeba/multi_clf -- multi=True
```
c) Train attribute classifier with UTKFace
```
python train_attribute_clf.py UTKFace ./results/UTKFace/attr_clf
```

#Prepare data-split
## 3) Generate the various Perc and bias spilits
The density ratio classifier should be trained for the appropriate `bias` and `perc` setting, which can be adjusted in the script below:
```
python get_density_ratios.py [celeba UTKFace] [ResNet18 CNN5 CNN3] --perc=[0.1, 0.25, 0.5, 1.0] --bias=[90_10, multi]
```
By employing `--ncf` argument, you can control the complexity of CNN classifiers.
Note that the best density ratio classifier will be saved in its corresponding directory under `./data/[celeba UTKFace FairFace]`, you should transfer this to `./data`.


#2) To Train the Baseline Model
## Use `./BIGGAN/src/Base_imp_weight` for (Choi etal)
> Use --reweight 1 for Choi et.al Importance-weights
> Select the --perc setting e.g. {0.25,0.1,0.05,0.025}
> Select the --bias settings e.g. {90_10,multi}, if you are testing multi you have to append --multi 1 for the model to reference the correct FD classifier
```
python train.py --shuffle --batch_size 128 --parallel --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset CA64 --data_root ../../data --G_ortho 0.0 --G_attn 0 \
--D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 2 --num_save_copies 1 \
--loss_type hinge --seed 777 --num_epochs 200 --start_eval 40 --reweight 1 --alpha 1.0 --bias 90_10 --perc 1.0 --name_suffix celeba_90_10_perc1.0_Baseline
```

#3) To Train the Pre-trained Model for fairTL++ and fairTL
## Use `./BIGGAN/src/FairGAN++` for  our proposed work
> Use --reweight 0 for regular traing on the reference + bias data
> Select the perc setting e.g. {0.25,0.1,0.05,0.025}
> Select the --bias settings e.g. {90_10,multi}, if you are testing multi you have to append --multi 1 for the model to reference the correct FD classifier
```
python train.py --shuffle --batch_size 128 --parallel --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 5e-4 --D_lr 2e-4 --dataset CA64 --data_root ../../data --G_ortho 0.0 --G_attn 0 \
--D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 2 --num_save_copies 1 \
--loss_type hinge --seed 777 --num_epochs 200 --start_eval 40 --reweight 0 --alpha 1.0 --bias 90_10 --perc 1.0 --name_suffix celeba_90_10_perc1.0_pretrained
```

#4) FairTL
## Use `./BIGGAN/src/FairGAN++` for  our proposed work
> To perform fairTL we select the new "./weights/###_copy0" pre-trained weights and copy them into a new file of your own naming choice e.g., celeba_90_10_perc1.0_pretrained_Linear_Prob
> Include --dummy 2 to train on the uniform D_ref dataset only 
```
python train.py --shuffle --batch_size 8 --parallel --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 5e-4 --D_lr 2e-4 --dataset CA64 --data_root ../../data --G_ortho 0.0 --G_attn 0 \
--D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 2 --num_save_copies 1 \
--loss_type hinge --seed 777 --num_epochs 300 --start_eval 40 --reweight 0 --alpha 1.0 --bias 60_40 --perc 1.0 --name_suffix celeba_90_10_perc1.0_pretrained\
--resumePaused --load_weight copy0 --dummy 2
```

#5) (FairTL++) Perform linear probing
## Use `./BIGGAN/src/FairGAN++` for  our proposed work
> To perform linear probing we select the new "./weights/###_copy0" pre-trained weights and copy them into a new file of your own naming choice e.g., celeba_90_10_perc1.0_pretrained_Linear_Prob
> Include --dummy 2 to train on the uniform D_ref dataset only 
```
python train_LP.py --shuffle --batch_size 8 --parallel --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 5e-4 --D_lr 2e-4 --dataset CA64 --data_root ../../data --G_ortho 0.0 --G_attn 0 \
--D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 2 --num_save_copies 1 \
--loss_type hinge --seed 777 --num_epochs 200 --start_eval 40 --reweight 0 --alpha 1.0 --bias 90_10 --perc 1.0 --name_suffix celeba_90_10_perc1.0_pretrained\
--resumePaused --load_weights=copy0 --dummy 2
```

#6) (FairTL++) Perform Fine Tuning
## Use `./BIGGAN/src/FairGAN++` for  our proposed work
> Similarly select the new "./weights/###_copy0" weights and copy them into a new file e.g., celeba_90_10_perc1.0_pretrained_Linear_Prob_Fine_tuning
> Include --dummy 2 to train on the uniform D_ref dataset only 
```
python train_FT.py --shuffle --batch_size 8 --parallel --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 5e-4 --D_lr 2e-4 --dataset CA64 --data_root ../../data --G_ortho 0.0 --G_attn 0 \
--D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 2 --num_save_copies 1 \
--loss_type hinge --seed 777 --num_epochs 200 --start_eval 40 --reweight 0 --alpha 1.0 --bias 90_10 --perc 1.0 --name_suffix celeba_90_10_perc1.0_pretrained\
--resumePaused --load_weights=copy0 --dummy 2
```



# b) Fair Generative Model Via Transfer Learning From StyleGANv2

#1) Preprocessing Raw data into the respective SA splits
> Base codes were adopted from karras et.al https://github.com/NVlabs/stylegan2-ada-pytorch and labels from https://github.com/DCGM/ffhq-features-dataset
> Download the FFHQ dataset from https://github.com/NVlabs/ffhq-dataset and place it in `./StyleGANv2/Prepared_data/data`
>Run the following pre-processing script in `./StyleGANv2/Prepared_data`. This will output 3 types of files 1) File for training the GAN e.g., .`/training/Gender/GenderTrainSamples_0.025/` 2)FID reference files e.g., .`/training/Gender/GenderFIDSamples/` 3)`Gender_FFHQ_Clf_Training_data.pt` and `Gender_FFHQ_Clf_Training_labels.pt`
```
python sort_data.py
```

#2) Preprocessing the data into the zip file StyleGAN2 code requires
>zip the file for training 
>Select the source data i.e., ./StyleGANv2/stylegan2-ada-pytorch-main/Prepare_data/training/Gender/GenderTrainSamples_0.025/
>Select the output dataset name (this can be anything you want)
```
python dataset_tool.py --source=*source of Dref* --dest=~/datasets/*dataset_name*.zip
```

#2) Train attribute classifier
>Use the directiory ./StyleGANv2/stylegan2-ada-pytorch-main//metrics/FD
>Then copy the data from 3)`Gender_FFHQ_Clf_Training_data.pt` and `Gender_FFHQ_Clf_Training_labels.pt` into ./StyleGANv2/stylegan2-ada-pytorch-main//metrics/FD/data/ and post process the data to train/Test/split with: 
```
python Prepare_training_data.py
```
then train the classifier with (Fill sensitive attribute(SA) name):
```
python train_attribute_clf.py -attribute *SA*
```


#3)Resume sensitive adaptation with styleGANv2
>Training the pre-trained styleGANv2 on the new Dref data with linear probing 
>Fill *training datafile* with output file from 2)
>Fill *network pickle file*
```
python train_2D_LP.py --outdir=~/training-runs --data=~/mydataset/*training datafile* --gpus=1 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl --aug=noaug --kimg 200
``` 
followed by fine-tuining:
>*network pickle file* is the output of the LP file
```
python train_2D.py --outdir=~/training-runs --data=~/mydataset/*training datafile* --gpus=1 --resume=*network pickle file* --aug=noaug --kimg 800
``` 

#4)Measuring Fairness and FID 
> Select *FID ref file* from 2)FID reference files e.g., .`/training/Gender/GenderFIDSamples/
> Select *saved network* from the training file
> Select *SA* to evaluate 
```
python cal_FD --attribute *SA* --network *saved network*
```
```
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/*FID ref file* --mirror=1 --network *saved network*
```