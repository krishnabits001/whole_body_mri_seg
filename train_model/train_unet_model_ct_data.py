
# coding: utf-8

# In[1]:

__author__ = 'krishnch'

import os
# # Assign GPU no
#os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']
#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import numpy as np

#to make directories
import pathlib

import sys
sys.path.append("../")

from utils import *


import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='ct_data', choices=[ 'ct_data', 'none'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr80', choices=['tr80', 'trall'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1', choices=['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10'])
#learning rate of unet
parser.add_argument('--lr_seg', type=float, default=0.001) 

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])

#run version 
parser.add_argument('--ver', type=int, default=0)

#dsc loss
parser.add_argument('--dsc_loss', type=int, default=2)
#num of input channels
parser.add_argument('--num_channels', type=int, default=1)
#wgt_fac weighted cross entropy
parser.add_argument('--wgt_fac', type=int, default=0)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])


# In[3]:

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
#define save_dir for the model
if(parse_config.data_aug==0):
    save_dir=str(cfg.base_dir)+'/ct_models/fat_vs_muscle/baseline_unet/no_data_aug/'+str(parse_config.no_of_tr_imgs)+         '/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_'+str(parse_config.wgt_fac)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
    cfg.aug_en=parse_config.data_aug
    print('cfg.aug_en',cfg.aug_en)
else:
    save_dir=str(cfg.base_dir)+'/ct_models/fat_vs_muscle/baseline_unet/with_data_aug/'+str(parse_config.no_of_tr_imgs)+         '/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_wgt_fac_'+str(parse_config.wgt_fac)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
print('save dir ',save_dir)
print('cfg aug_en, num_classes ',cfg.aug_en,cfg.num_classes)
#sys.exit()
######################################

######################################
# load train and val images
train_list=[]
#1 to 80
for train_index in range(1,81):
    train_list.append(str(train_index))

print('loading train imgs')
print('train list',train_list)

train_imgs,train_labels = load_imgs(dt=dt,orig_img_dt=orig_img_dt,test_list=train_list)

print(train_imgs.shape,train_labels.shape)
print(np.min(train_imgs),np.max(train_imgs),np.mean(train_imgs))


val_list=[]
#81 to 90
for val_index in range(81,91):
    val_list.append(str(val_index))

print('loading val imgs')
print('val_list',val_list)

#load both val data and its cropped images
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)


# get test list
print('get test imgs list')
test_list=[]
#91 to 100
for test_index in range(91,101):
    test_list.append(str(test_index))
    
print('test list',test_list)
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################


######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
#checkpoint_filename='unet_'+str(parse_config.dataset)+'_nlabels_'+str(cfg.num_classes)
checkpoint_filename='unet_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################

######################################
#  training parameters
start_epoch=0
n_epochs = 10000
disp_step= 250
mean_f1=0
mean_f1_val_prev=0.05
threshold_f1=0.0005
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################

######################################
# define graph
tf.reset_default_graph()
print('cfg.dsc_loss',parse_config.dsc_loss)
ae = model.unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,num_channels=parse_config.num_channels,                wgt_fac=parse_config.wgt_fac)
######################################


######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

######################################
# create a session and initialize variable to use the graph
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
# Save training data
saver = tf.train.Saver(max_to_keep=2)
######################################


#####################
#debug
#start_epoch,n_epochs=0,301
#disp_step=50
#mean_f1=0
#####################


# Fit all training data in n_epochs
for epoch_i in range(start_epoch,n_epochs):

    #sample Labelled data shuffled batch
    ld_img_batch,ld_label_batch=shuffle_minibatch([train_imgs,train_labels],batch_size=cfg.batch_size,num_channels=parse_config.num_channels,axis=2)
    #ld_img_batch,ld_label_batch=shuffle_minibatch_nk([train_imgs,train_labels],num_labels=num_labels,batch_size=cfg.batch_size,num_channels=parse_config.num_channels,axis=2)
    #print("labelled data shape",ld_img_batch.shape,ld_label_batch.shape)
    if(cfg.aug_en==1):
        ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)

    #Optimer on this batch
    train_summary,_ =sess.run([ae['train_summary'],ae['optimizer_unet_seg']],feed_dict={ae['x']: ld_img_batch, ae['y_l']: ld_label_batch,
                                         ae['train_phase']: True})

    if ((epoch_i%disp_step == 0) or (epoch_i==n_epochs-1)):
        total_cost=sess.run(ae['seg_cost'], feed_dict={ae['x']: ld_img_batch, ae['y_l']: ld_label_batch,                                                       ae['train_phase']: False})
        #print("ep, total cost",epoch_i,total_cost)
        print("ep, total cost, mean_f1",epoch_i,total_cost, mean_f1)


    if(epoch_i%val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

    if(epoch_i%val_step_update==0):
        ##Save the model with best DSC for Validation Image
        mean_f1_arr=[]
        f1_arr=[]
        mean_total_cost_val=0
        for val_id_no in range(0,len(val_list)):
            val_img_crop_tmp=val_img_crop[val_id_no]
            val_label_crop_tmp=val_label_crop[val_id_no]
            val_label_orig_tmp=val_label_orig[val_id_no]
            pixel_size_val=pixel_val_list[val_id_no]

            pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, val_img_crop_tmp)
            re_pred_mask_sys,f1_val = f1_util.reshape_img_and_f1_score(pred_sf_mask, val_label_orig_tmp, pixel_size_val)
            mean_f1_arr.append(np.mean(f1_val[1:cfg.num_classes]))
            f1_arr.append(f1_val[1:cfg.num_classes])


        #avg mean over 2 val subjects
        mean_f1_arr=np.asarray(mean_f1_arr)
        #print('mean f1 arr',mean_f1_arr)
        mean_f1=np.mean(mean_f1_arr)
        #print('mean f1',mean_f1)
        f1_arr=np.asarray(f1_arr)
        #print('f1 arr',f1_arr,f1_arr.shape,f1_arr[:,0])

        if (mean_f1-mean_f1_val_prev>threshold_f1):
            print("prev f1_val; present_f1_val, array", mean_f1_val_prev, mean_f1, mean_f1_arr)
            mean_f1_val_prev = mean_f1
            #directly save the model
            print("best model saved at epoch no. ", epoch_i)
            mp_best = str(best_model_dir) + str(checkpoint_filename) + '_best_model_epoch_' + str(epoch_i) + ".ckpt"
            saver.save(sess, mp_best)

        #calc. and save validation image dice summary
        dsc_summary_msg = sess.run(ae['val_dsc_summary'], feed_dict={ae['mean_dice']: mean_f1})
        val_sum_writer.add_summary(dsc_summary_msg, epoch_i)
        val_sum_writer.flush()

    if ((epoch_i==n_epochs-1) and (epoch_i != start_epoch)):
        print("model saved at epoch no. ", epoch_i)
        # Saving learnt CNN Model architecture and hyperparamters into checkpoint file
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp


sess.close()

######################################
# restore best model and predict segmentations on test subjects
saver_new = tf.train.Saver()
sess_new = tf.Session(config=config)
saver_new.restore(sess_new, mp_best)
print("best model chkpt",mp_best)
print("Model restored")

f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir,orig_img_dt,test_list,struct_name,only_segnet=1)

save_dir_tmp=str(save_dir)+'/train_imgs_dsc/'
test_list=train_list
f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,test_list,struct_name,only_segnet=1)

save_dir_tmp=str(save_dir)+'/val_imgs_dsc/'
test_list=val_list
f1_util.pred_segs_bodyfat_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,test_list,struct_name,only_segnet=1)
######################################


