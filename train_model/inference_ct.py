__author__ = 'krishnch'

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
parser.add_argument('--dataset', type=str, default='ct_data', choices=['ct_data','none'])

#parser.add_argument('--ip_path', type=str, default=None)
#CT image
parser.add_argument('--ip_path_ct', type=str, default='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/new_scan_data/ct/sl_im/p92.nii.gz')

#version of run
#parser.add_argument('--out_path', type=str, default=None)
parser.add_argument('--out_path', type=str, default='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/output/ct/')

#patient id
parser.add_argument('--patient_id', type=str, default='92')

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

if parse_config.dataset == 'ct_data':
    print('load diff. fat configs')
    import experiment_init.ct_init_fat_vs_muscle_tissues as cfg
    import experiment_init.ct_data_cfg_bodyfat as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'ct_data':
    print('set bodyfat orig img dataloader handle')
    orig_img_dt=dt.load_ct_fat_img_labels

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################
#define image path
img_path_ct=str(parse_config.ip_path_ct)

#checking if image path exists or not
if(os.path.isfile(img_path_ct)==False):
    print('img path for CT image does not exist')
    sys.exit()

#segmentation output path directory 
# by default same as input directory unless defined
out_path = str(parse_config.out_path)
if(out_path=='None'):
    #input directory path
    print('out dir same as input dir')
    out_path = os.path.dirname(parse_config.ip_path)

######################################
#define save_dir for the model
#1 channel CT image model
save_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/ct_models/fat_vs_muscle/baseline_unet/with_data_aug/tr80/c1_v0/'+        'unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_0_lr_seg_0.001/'

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
######################################

######################################
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

######################################
#Load CT image
ct_img_orig,affine_tst_ct,pixel_size = load_img(img_path_ct)

#expand axis if it has only 2 dimensions (2D slice)
if(len(ct_img_orig.shape)==2):
    ct_img_orig = np.expand_dims(ct_img_orig,axis=2)

# Normalize input data using min-max normalization
ct_img_norm=dt.normalize_minmax_data(ct_img_orig)

#dummy labels with zeros in native resolution
label_sys=np.zeros_like(ct_img_norm)

#re-sample & crop into pre-defined in-plane resolution & dimensions, respectively
cropped_ct_img = dt.preprocess_data(ct_img_norm, label_sys, pixel_size, label_present=0)
crop_ct_img = change_axis_img([cropped_ct_img],labels_present=0)

######################################
print('Loaded image(s) & inferring predicted mask')
#Find the predicted mask for the input image(s) on the Trained model
pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, crop_ct_img, batch_factor=40)

final_predicted_mask,_ = f1_util.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)
######################################

######################################

print('Saving precited mask...')
#save segmentation mask in nifti format
array_mask = nib.Nifti1Image(final_predicted_mask.astype(np.int16), affine_tst_ct)

out_path_tmp = out_path + '/ct_images_model/patient_id_'+str(parse_config.patient_id)+'/'
pathlib.Path(out_path_tmp).mkdir(parents=True, exist_ok=True)

pred_filename = str(out_path_tmp)+'/pred_mask.nii.gz'

#Also, save ct image in nifti format
pred_filename_img_ct= str(out_path_tmp)+'ct_image.nii.gz'
array_img_ct = nib.Nifti1Image(ct_img_orig, affine_tst_ct)

nib.save(array_img_ct, pred_filename_img_ct)
    
nib.save(array_mask, pred_filename)

######################################
