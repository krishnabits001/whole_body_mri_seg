__author__ = 'krishnch'

import os
# # Assign GPU no
# os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']
# from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import time

#to make directories
import pathlib

import sys
sys.path.append("../")

from utils import *

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='diff_fat', choices=['diff_fat', 'none'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr8', choices=['tr8', 'trall'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1', choices=['c1', 'c2'])
#learning rate of unet
parser.add_argument('--lr_seg', type=float, default=0.001) 

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--ver', type=int, default=0)

#dsc loss
parser.add_argument('--dsc_loss', type=int, default=0)
#wgt_fac
parser.add_argument('--wgt_fac', type=int, default=0)

parse_config = parser.parse_args()

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

######################################
#define save_dir for the model
if(parse_config.data_aug==0):
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/baseline_unet/no_data_aug/'+str(parse_config.no_of_tr_imgs)+\
         '/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_'+str(parse_config.wgt_fac)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
    cfg.aug_en=parse_config.data_aug
    print('cfg.aug_en',cfg.aug_en)
else:
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/baseline_unet/with_data_aug/'+str(parse_config.no_of_tr_imgs)+\
         '/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_'+str(parse_config.wgt_fac)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
print('save dir ',save_dir)
print('cfg aug_en, num_classes ',cfg.aug_en,cfg.num_classes)
######################################

######################################
# load train and val images
if(parse_config.no_of_tr_imgs=='trall'):
    train_list = data_list.tf_train_data()
else:
    train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#print(train_list)
#train_imgs,train_labels = load_imgs(dt=dt,orig_img_dt=orig_img_dt,test_list=train_list)

#validation images list
val_list = data_list.val_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
print('loading val imgs')
print('val list',val_list)

# get test list
print('get test imgs list')
test_list = data_list.test_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
print('test list',test_list)
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################

######################################
# define graph
tf.reset_default_graph()
print('cfg.dsc_loss',parse_config.dsc_loss)
ae = model.unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,wgt_fac=parse_config.wgt_fac)
######################################

print('save dir ',save_dir)
mp_best=get_max_chkpt_file(save_dir)
print('load mp',mp_best)

#######################################
# restore best model and predict segmentations on test subjects
saver_new = tf.train.Saver()
sess_new = tf.Session(config=config)
saver_new.restore(sess_new, mp_best)
print("best model chkpt",mp_best)
print("Model restored")

#predict segmentations on test subjects
f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir,orig_img_dt,test_list,struct_name,only_segnet=1)

#predict segmentations on training subjects
save_dir_tmp=str(save_dir)+'/train_imgs_dsc/'
test_list=train_list
f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,test_list,struct_name,only_segnet=1)

#predict segmentations on validation subjects
save_dir_tmp=str(save_dir)+'/val_imgs_dsc/'
test_list=val_list
f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,test_list,struct_name,only_segnet=1)
######################################
