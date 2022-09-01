#!/bin/bash

cd <git_repo>/train_model/

# activate tensorflow environment (env_name)
source activate <env_name>

#inference for any new test image - only fat image - 1 channel input
python inference_mri.py --ip_path_fat=<path_of_input_fat_image> --out_path=<path_to_save_predicted_segmentation_mask>

#inference for any new test image - both fat & water images as 2 channel input
#python inference_mri.py --ip_path_fat=<path_of_input_fat_image>  --ip_path_water=<path_of_input_water_image> --out_path=<path_to_save_predicted_segmentation_mask>

#to run inference on train, test and validation sets used for model training;
#model trained with affine augmentations
#python inference_on_trained_unet.py --no_of_tr_imgs=tr8 --comb_tr_imgs=c4 --data_aug=1 --dsc_loss=0 --ver=0
