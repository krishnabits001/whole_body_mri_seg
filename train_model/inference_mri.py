# coding: utf-8

import os
# # Assign GPU no
# os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

#to make directories
import pathlib
import nibabel as nib

import sys
sys.path.append('../')

from utils import *

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='diff_fat', choices=['diff_fat','none'])

#parser.add_argument('--ip_path', type=str, default=None)
#fat image
#parser.add_argument('--ip_path_fat', type=str, default='/usr/bmicnas01/data-biwi-01/krishnch/datasets/bodyfat_uzh/orig/004/fat_img.nii.gz')
parser.add_argument('--ip_path_fat', type=str, default='/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/bodyfat_orig/top_down_nifti/top_down_niftyp10_z1_f.nii.gz')
#water image
#parser.add_argument('--ip_path_water', type=str, default='/usr/bmicnas01/data-biwi-01/krishnch/datasets/bodyfat_uzh/orig_cut/004/water_img.nii.gz')
#parser.add_argument('--ip_path_water', type=str, default='/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/bodyfat_orig/orig/004/water_img.nii.gz')
parser.add_argument('--ip_path_water', type=str, default=None)

#version of run
#parser.add_argument('--out_path', type=str, default=None)
parser.add_argument('--out_path', type=str, default='/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/bodyfat_orig/top_down_nifti/')

#patient id
parser.add_argument('--patient_id', type=str, default='10')

#time point
parser.add_argument('--time_pt', type=str, default='1')

#enable 1-hot encoding of the labels
parser.add_argument('--en_1hot', type=int, default=0)
#dsc loss
parser.add_argument('--dsc_loss', type=int, default=2)
#wgt_fac weighted cross entropy
parser.add_argument('--wgt_fac', type=int, default=0)
#num of input channels
parser.add_argument('--num_channels', type=int, default=1)
#learning rate of unet
parser.add_argument('--lr_seg', type=float, default=0.001)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'diff_fat':
    print('load diff. fat configs')
    import experiment_init.init_sub_vs_visc_fat as cfg
    import experiment_init.data_cfg_bodyfat as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'diff_fat':
    print('set bodyfat orig img dataloader handle')
    orig_img_dt=dt.load_diff_fat_types_img_labels

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

#define image path
img_path_fat=str(parse_config.ip_path_fat)
img_path_water=str(parse_config.ip_path_water)

#checking if image path exists or not
if(os.path.isfile(img_path_fat)==False):
    print('img path for fat image does not exist')
    sys.exit()

if(os.path.isfile(img_path_water)==False):
    print('img path for water image does not exist')
    #sys.exit()
    print('chosing 1 channel (FAT image) inference model')
    parse_config.num_channels = 1
else:
    print('chosing 2 channels (FAT + Water images) inference model')
    parse_config.num_channels = 2
    
#segmentation output path directory 
# by default same as input directory unless defined
out_path = str(parse_config.out_path)
if(out_path=='None'):
    #input directory path
    print('out dir same as input dir')
    out_path = os.path.dirname(parse_config.ip_path)

######################################
#define save_dir for the model
#save_dir='../../bodyfat_seg/models/diff_fat/baseline_unet/with_data_aug/tr8/'
if(parse_config.num_channels==1):
    #1 channel (fat) image model
    save_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/git_repo_jun18_2020/tr_models_final/models/diff_fat/only_fat_img/baseline_unet/with_data_aug/tr8/c4_v0/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_0_lr_seg_0.001/'
else:
    #2 channels (fat + water) model
    save_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/git_repo_jun18_2020/tr_models_final/models/diff_fat/baseline_unet/with_data_aug/tr8/c4_v0/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_0_lr_seg_0.001/'

#save_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/trained_models_final/models/diff_fat/baseline_unet/with_data_aug/tr8/c4_v0/unet_dsc_0_wgt_fac_0_lr_seg_0.001/'

print('save dir ',save_dir)

#find the model with best dice score on validation images
mp_best=get_max_chkpt_file(save_dir)
print('load mp',mp_best)

######################################
# define U-Net model graph
tf.reset_default_graph()
if(parse_config.en_1hot):
    ae = model.unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,num_channels=parse_config.num_channels,                    mixup_en=parse_config.en_1hot)
else:
    ae = model.unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,num_channels=parse_config.num_channels)

# restore best model and predict segmentations on test subjects
saver = tf.train.Saver()
sess = tf.Session(config=config)
saver.restore(sess, mp_best)
print("Model restored")
######################################

def load_img(img_path):
    # Load the input image
    image_data_test_load = nib.load(img_path)
    image_data_test_sys=image_data_test_load.get_data()
    pixel_size=image_data_test_load.header['pixdim'][1:4]
    affine_tst=image_data_test_load.affine
    
    return image_data_test_sys,affine_tst,pixel_size


#Load Fat image
fat_img_orig,affine_tst_fat,pixel_size = load_img(img_path_fat)

if(parse_config.num_channels==2):
    #Load Water image
    water_img_orig,affine_tst_water,pixel_size = load_img(img_path_water)

# Normalize input data using min-max normalization
fat_img_norm=dt.normalize_minmax_data(fat_img_orig)
if(parse_config.num_channels==2):
    water_img_norm=dt.normalize_minmax_data(water_img_orig)

#dummy labels with zeros in native resolution
label_sys=np.zeros_like(fat_img_norm)

#re-sample & crop into pre-defined in-plane resolution & dimensions, respectively
cropped_fat_img = dt.preprocess_data(fat_img_norm, label_sys, pixel_size, label_present=0)
crop_fat_img = change_axis_img([cropped_fat_img],labels_present=0)

if(parse_config.num_channels==2):
    cropped_water_img = dt.preprocess_data(water_img_norm, label_sys, pixel_size, label_present=0)
    crop_water_img = change_axis_img([cropped_water_img],labels_present=0)

print('Loaded image(s) & inferring predicted mask')
#Find the predicted mask for the input image(s) on the Trained model
if(parse_config.num_channels==2):
    pred_sf_mask = f1_util.calc_pred_sf_mask_full_2channel(sess, ae, crop_fat_img, crop_water_img, batch_factor=40)
else:
    pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, crop_fat_img, batch_factor=40)


final_predicted_mask,_ = f1_util.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)

print('Saving precited mask...')
print('patient_id, time_pt',parse_config.patient_id,parse_config.time_pt)
#save segmentation mask in nifti format
array_mask = nib.Nifti1Image(final_predicted_mask.astype(np.int16), affine_tst_fat)
if(parse_config.num_channels==2):
    out_path_tmp = out_path + '/mri/fat_water_images_model/patient_id_'+str(parse_config.patient_id)+'/'
    pathlib.Path(out_path_tmp).mkdir(parents=True, exist_ok=True)
    pred_filename = str(out_path_tmp)+'pred_mask_timept_'+str(parse_config.time_pt)+'.nii.gz'
    
    #Also, save fat & water images in nifti format
    pred_filename_img_fat= str(out_path_tmp)+'fat_image_timept_'+str(parse_config.time_pt)+'.nii.gz'
    pred_filename_img_water= str(out_path_tmp)+'water_image_timept_'+str(parse_config.time_pt)+'.nii.gz'
    
    array_img_fat = nib.Nifti1Image(fat_img_orig, affine_tst_fat)
    array_img_water = nib.Nifti1Image(water_img_orig, affine_tst_water)
    
    nib.save(array_img_water, pred_filename_img_water)
    
else:
    out_path_tmp = out_path + '/mri/only_fat_image_model/patient_id_'+str(parse_config.patient_id)+'/'
    pathlib.Path(out_path_tmp).mkdir(parents=True, exist_ok=True)
    
    pred_filename = str(out_path_tmp)+'/pred_mask_timept_'+str(parse_config.time_pt)+'.nii.gz'
    
    #Also, save fat image in nifti format
    pred_filename_img_fat= str(out_path_tmp)+'fat_image_timept_'+str(parse_config.time_pt)+'.nii.gz'
    array_img_fat = nib.Nifti1Image(fat_img_orig, affine_tst_fat)

nib.save(array_img_fat, pred_filename_img_fat)
    
nib.save(array_mask, pred_filename)



