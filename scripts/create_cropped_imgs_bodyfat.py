__author__ = 'krishnch'

import numpy as np
import pathlib

import sys
sys.path.append("../")
import os.path
import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='diff_fat', choices=['diff_fat', 'none'])

parse_config = parser.parse_args()

if parse_config.dataset == 'diff_fat':
    print('load diff. fat configs')
    import experiment_init.init_sub_vs_visc_fat as cfg
    import experiment_init.data_cfg_bodyfat as data_list
else:
    raise ValueError(parse_config.dataset)

from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)
print(cfg)

if parse_config.dataset == 'diff_fat':
    print('set bodyfat orig img dataloader handle')
    orig_img_dt=dt.load_diff_fat_types_img_labels

#save dir define
# Load each image
study_list=['001']#,'002','004','005','006','007','010','011','013','015']

for index in study_list:
    test_id=str(index)
    test_id_l=[test_id]
    
    img_path='/usr/bmicnas01/data-biwi-01/krishnch/datasets/bodyfat_uzh/orig/'+str(test_id)+'/fat_img.nii.gz'
    
    if(os.path.isfile(img_path)):    
        #BodyFat UZH data
        print('study_id',test_id)
        img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
        print('b shape',test_id,img_sys.shape,label_sys.shape,pixel_size)
        if(pixel_size[2]==3):
            cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
            print('a shape',cropped_img_sys.shape,cropped_mask_sys.shape)

            save_dir_tmp=str(cfg.data_path_tr_cropped)+'/'+str(test_id)+'/'
            pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)
            print(save_dir_tmp)
            savefile_name=str(save_dir_tmp)+'img_cropped.npy' 
            np.save(savefile_name,cropped_img_sys)
            savefile_name=str(save_dir_tmp)+'mask_cropped.npy' 
            np.save(savefile_name,cropped_mask_sys)


