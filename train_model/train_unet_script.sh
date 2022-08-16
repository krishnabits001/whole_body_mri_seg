#!/bin/bash

cd <git_repo>/train_model/

# activate tensorflow environment (env_name)
source activate <env_name>

#train with affine augmentations
python train_baseline_unet.py --dataset=diff_fat --no_of_tr_imgs=tr8 --comb_tr_imgs=c4 --data_aug=1 --dsc_loss=0 --ver=0

#train with affine augmentations + random deformations and contrast augmentations
#python train_baseline_unet_with_rand_deformations.py --dataset=diff_fat --no_of_tr_imgs=tr8 --comb_tr_imgs=c4 --data_aug=1 --dsc_loss=0 --ver=0 --ri_en=1

