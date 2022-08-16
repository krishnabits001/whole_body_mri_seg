#!/bin/bash

cd <git_repo>/train_model/

# activate tensorflow environment (env_name)
source activate <env_name>

#inference for any new test image
python inference.py --ip_path=<path_of_input_image> --out_path=<path_to_save_predicted_segmentation_mask>

#to run inference on train, test and validation sets used for model training;
#model trained with affine augmentations
#python inference_on_trained_unet.py --no_of_tr_imgs=tr8 --comb_tr_imgs=c4 --data_aug=1 --dsc_loss=0 --ver=0
