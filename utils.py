__author__ = 'krishnch'

import numpy as np
import scipy.misc
import scipy.ndimage.interpolation

from skimage import transform
import random

import os
import re

def augmentation_function(ip_list, dt, labels_present=1, rescale_func=0, en_1hot=0):
    '''
    Function for augmentation of minibatches. It will transform a set of images and corresponding labels
    by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
    with random parameters.
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size.
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction.
    :return: A mini batch of the same size but with transformed images and masks.
    '''

    if(len(ip_list)==2 and labels_present==1):
        images = ip_list[0]
        labels = ip_list[1]
    else:
        images=ip_list[0]

    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for index in range(num_images):

        img = np.squeeze(images[index,...])
        if(labels_present==1):
            lbl = np.squeeze(labels[index,...])

        do_rotations,do_scaleaug,do_fliplr,do_simple_rot=0,0,0,0
        aug_select = np.random.randint(5)

        if(np.max(img)>0.001):
            if(aug_select==0):
                do_rotations=1
            elif(aug_select==1):
                do_scaleaug=1
            elif(aug_select==2):
                do_fliplr=1
            elif(aug_select==3):
                do_simple_rot=1

        if(rescale_func==1):
            do_scaleaug=0

        # ROTATE - random angle b/w -15 and 15
        if do_rotations:
            angles = [-15,15]
            random_angle = np.random.uniform(angles[0], angles[1])
            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM CROP SCALE
        if do_scaleaug:
            n_x, n_y = img.shape
            #scale factor between 0.95 and 1.05
            scale_fact_min=0.95
            scale_fact_max=1.05
            scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
            slice_rescaled = transform.rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant')
            img = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)
            if(labels_present==1):
                if(en_1hot==1):
                    slice_rescaled = transform.rescale(lbl, scale_val, order=1, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size_1hot(slice_rescaled, n_x, n_y)
                else:
                    slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                if(labels_present==1):
                    lbl = np.fliplr(lbl)

        # Simple rotations at angles of 45 degrees
        if do_simple_rot:
            fixed_angle = 180 #45
            random_angle = np.random.randint(3)*fixed_angle

            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        new_images.append(img[..., np.newaxis])
        if(labels_present==1):
            new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    if(labels_present==1):
        sampled_label_batch = np.asarray(new_labels)

    if(len(ip_list)==2 and labels_present==1):
        return sampled_image_batch, sampled_label_batch
    else:
        return sampled_image_batch

def calc_deform(cfg, mu=0,sigma=10, order=3):

    flow_vec = np.zeros((cfg.batch_size,cfg.img_size_x,cfg.img_size_y,2))

    for i in range(cfg.batch_size):
        #mu, sigma = 0, 1 # mean and standard deviation
        dx = np.random.normal(mu, sigma, 9)
        dx_mat = np.reshape(dx,(3,3))
        dx_img = transform.resize(dx_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')

        dy = np.random.normal(mu, sigma, 9)
        dy_mat = np.reshape(dy,(3,3))
        dy_img = transform.resize(dy_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')


        flow_vec[i,:,:,0] = dx_img
        flow_vec[i,:,:,1] = dy_img
    #print(flow_vec.shape)

    return flow_vec

def shuffle_minibatch_nk(ip_list,num_labels, batch_size=20,num_channels=1,labels_present=1,axis=2):
    """
    batch_size number of 2D slices are randomly sampled from the total 3D MRI scan data.
    This can be 1/30 concatenated training images as per the input provided.
    Corresponding 2D slices indices of the labels are selected as well.
    These are used for each training iteration of the CNN where in each
    iteration we get batch_size no of 2D slice samples randomly choosen from
    the total input data. These 2D slices differ for each iteration hence
    improves the robustness of the CNN trained.

    num_channels = 1 and num_classes = 3 for this prostate data.
    Args:
        image_data_train : 3D MRI scan data out of which batch_size number of
                           slices are selected and used for the training of CNN.
        label_data_train : 3D MRI scan labels data indices corresponding to test
                           data that are selected and used for the training of CNN.
        batch_size : number of 2D slices to consider for the training of CNN.
        num_channels : no of channels of the input image
        axis : the axis along which we want to sample the minibatch -> axis vals : 0 - for sagittal, 1 - for coronal, 2 - for axial
    Returns:
        image_data_train_batch : concatenated 2D slices randomly choosen from the total input data.
        label_data_train_batch : concatenated 2D slices of labels with indices corresponding to the input data selected.
    """
    if(len(ip_list)==2 and labels_present==1):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
    else:
        image_data_train=ip_list[0]

    img_size_x=image_data_train.shape[0]
    img_size_y=image_data_train.shape[1]
    img_size_z=image_data_train.shape[2]

    len_of_train_data=np.arange(image_data_train.shape[axis])
    # set how many images of the respective label number should be chosen
    label_weights = [2, 4, 9, 5]

    count = 0
    for lw in range(4):
        label_indices = np.where(num_labels==(lw+1))[0]
        # print("label_indices: {}, nr of images to pick: {}".format(
        #    len(label_indices), label_weights[lw]))
        random_indices = np.random.choice(label_indices,
                                           size=label_weights[lw],
                                          replace=True)
        for index_no in random_indices:
            img_train_tmp = np.reshape(image_data_train[:, :, index_no],
                                       (1, img_size_x, img_size_y, num_channels))
            label_train_tmp = np.reshape(label_data_train[:, :, index_no],
                                         (1, img_size_x, img_size_y))
            # print("shape of tmp: {}".format(img_train_tmp.shape))
            #reg_target_tmp = calculate_reg_target(label_train_tmp)
            #reg_target_tmp = np.swapaxes(reg_target_tmp, axis1=0, axis2=1)
            if count == 0:
                image_data_train_batch=img_train_tmp
                label_data_train_batch=label_train_tmp
                #reg_targets_train_batch=reg_target_tmp
            else:
                image_data_train_batch = np.concatenate(
                    (image_data_train_batch, img_train_tmp), axis=0)
                label_data_train_batch = np.concatenate(
                    (label_data_train_batch, label_train_tmp), axis=0)
                #reg_targets_train_batch = np.concatenate(
                #    (reg_targets_train_batch, reg_target_tmp), axis=0)
            count += 1


    if(len(ip_list)==2 and labels_present==1):
        #return image_data_train_batch, label_data_train_batch, reg_targets_train_batch
        return image_data_train_batch, label_data_train_batch
    else:
        return image_data_train_batch

def shuffle_minibatch(ip_list, batch_size=20,num_channels=1,labels_present=1,axis=2):
    """
    batch_size number of 2D slices are randomly sampled from the total 3D MRI scan data. This can be 1/30 concatenated training images as per the input provided.
    Corresponding 2D slices indices of the labels are selected as well.
    These are used for each training iteration of the CNN where in each iteration we get batch_size no of 2D slice samples randomly choosen from the total input data. These 2D slices differ for each iteration hence improves the robustness of the CNN trained.
    num_channels = 1 and num_classes = 3 for this prostate data.
    Args:
        image_data_train : 3D MRI scan data out of which batch_size number of slices are selected and used for the training of CNN.
        label_data_train : 3D MRI scan labels data indices corresponding to test data that are selected and used for the training of CNN.
        batch_size : number of 2D slices to consider for the training of CNN.
        num_channels : no of channels of the input image
        axis : the axis along which we want to sample the minibatch -> axis vals : 0 - for sagittal, 1 - for coronal, 2 - for axial
    Returns:
        image_data_train_batch : concatenated 2D slices randomly choosen from the total input data.
        label_data_train_batch : concatenated 2D slices of labels with indices corresponding to the input data selected.
    """
    if(len(ip_list)==2 and labels_present==1):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
    else:
        image_data_train=ip_list[0]

    img_size_x=image_data_train.shape[0]
    img_size_y=image_data_train.shape[1]
    img_size_z=image_data_train.shape[2]

    len_of_train_data=np.arange(image_data_train.shape[axis])

    randomize=np.random.choice(len_of_train_data,size=len(len_of_train_data),replace=True)

    count=0
    for index_no in randomize:
        if(axis==2):
            img_train_tmp=np.reshape(image_data_train[:,:,index_no],(1,img_size_x,img_size_y,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,:,index_no],(1,img_size_x,img_size_y))
        elif(axis==1):
            img_train_tmp=np.reshape(image_data_train[:,index_no,:,],(1,img_size_x,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,index_no,:],(1,img_size_x,img_size_z))
        else:
            img_train_tmp=np.reshape(image_data_train[index_no,:,:],(1,img_size_y,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[index_no,:,:],(1,img_size_y,img_size_z))

        if(count==0):
            image_data_train_batch=img_train_tmp
            if(labels_present==1):
                label_data_train_batch=label_train_tmp
        else:
            image_data_train_batch=np.concatenate((image_data_train_batch, img_train_tmp),axis=0)
            if(labels_present==1):
                label_data_train_batch=np.concatenate((label_data_train_batch, label_train_tmp),axis=0)
        count=count+1
        if(count==batch_size):
            break

    if(len(ip_list)==2 and labels_present==1):
        return image_data_train_batch, label_data_train_batch
    else:
        return image_data_train_batch

def change_axis_img(ip_list, labels_present=1, def_axis_no=2, cat_axis=0):
    # Swap axes of 3D volume to easy computation of dice and other scores
    if(len(ip_list)==2 and labels_present==1):
        labeled_data_imgs = ip_list[0]
        labeled_data_labels = ip_list[1]
    else:
        labeled_data_imgs=ip_list[0]

    #can also define in a init file - base values
    img_size_x=labeled_data_imgs.shape[0]
    img_size_y=labeled_data_imgs.shape[1]

    total_slices = labeled_data_imgs.shape[def_axis_no]
    for slice_no in range(total_slices):

        img_test_slice = np.reshape(labeled_data_imgs[:, :, slice_no], (1, img_size_x, img_size_y, 1))
        if(labels_present==1):
            label_test_slice = np.reshape(labeled_data_labels[:, :, slice_no], (1, img_size_x, img_size_y))

        if (slice_no == 0):
            mergedlist_img = img_test_slice
            if(labels_present==1):
                mergedlist_labels = label_test_slice

        else:
            mergedlist_img = np.concatenate((mergedlist_img, img_test_slice), axis=cat_axis)
            if(labels_present==1):
                mergedlist_labels = np.concatenate((mergedlist_labels, label_test_slice), axis=cat_axis)

    if(len(ip_list)==2 and labels_present==1):
        return mergedlist_img,mergedlist_labels
    else:
        return mergedlist_img


def load_imgs(dt,orig_img_dt,test_list):

    count=0
    for test_id in test_list:
        test_id_l=[test_id]
        #load image,label pairs and process it to chosen resolution and dimensions
        img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
        cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
 
        #change axis for quicker computation of dice score
        #cropped_img_sys,cropped_mask_sys= change_axis_img([cropped_img_sys,cropped_mask_sys])
        #print('c',cropped_img_sys.shape)
 
        if(count==0):
            merged_cropped_imgs=cropped_img_sys
            merged_cropped_masks=cropped_mask_sys
            count=1
        else:
            merged_cropped_imgs=np.concatenate((merged_cropped_imgs,cropped_img_sys),axis=2)
            merged_cropped_masks=np.concatenate((merged_cropped_masks,cropped_mask_sys),axis=2)
    
    return merged_cropped_imgs,merged_cropped_masks

#load val images
def load_val_imgs(val_list,dt,orig_img_dt):
    #list for acdc - ES imgs
    val_label_orig_es=[]
    val_img_re_es=[]
    val_label_re_es=[]
    pixel_val_list=[]

    for val_id in val_list:
        val_id_list=[val_id]
        #val_img,val_label,pixel_size_val=dt.load_fat_img_labels(val_id_list)
        #print("0b val data shape",val_img.shape,val_label.shape)
        val_img,val_label,pixel_size_val=orig_img_dt(val_id_list)
        #print("1b val data shape",val_img.shape,val_label.shape)
        val_cropped_img,val_cropped_mask = dt.preprocess_data(val_img, val_label, pixel_size_val)
        #print("es val data shape",val_cropped_img.shape,val_cropped_mask.shape)

        #change axis for quicker computation of dice score
        val_img_re,val_labels_re= change_axis_img([val_cropped_img,val_cropped_mask])
        #print('re',val_img_re.shape, val_labels_re.shape)

        val_label_orig_es.append(val_label)
        val_img_re_es.append(val_img_re)
        val_label_re_es.append(val_labels_re)
        pixel_val_list.append(pixel_size_val)

    return val_label_orig_es,val_img_re_es,val_label_re_es,pixel_val_list

def get_max_chkpt_file(model_path,min_ep=10):
    for dirName, subdirList, fileList in os.walk(model_path):
        fileList.sort()
        #min_ep=10
        #print(fileList)
        for filename in fileList:
            if "meta" in filename.lower() and 'best_model' in filename:
                numbers = re.findall('\d+',filename)
                #print('model_path',model_path,filename)
                #print('0',filename,numbers,numbers[0],min_ep)
                if "_v2" in filename:
                    tmp_ep_no=int(numbers[1])
                else:
                    tmp_ep_no=int(numbers[0])
                if(tmp_ep_no>min_ep):
                    chkpt_max=os.path.join(dirName,filename)
                    min_ep=tmp_ep_no
    #print(chkpt_max)
    fin_chkpt_max = re.sub('\.meta$', '', chkpt_max)
    #print(fin_chkpt)
    return fin_chkpt_max

def isNotEmpty(s):
    return bool(s and s.strip())

def get_chkpt_file(model_path,match_name='',min_ep=10):
    for dirName, subdirList, fileList in os.walk(model_path):
        fileList.sort()
        #min_ep=10
        #print(fileList)
        for filename in fileList:
            if "meta" in filename.lower():
                numbers = re.findall('\d+',filename)
                #print('model_path',model_path,filename)
                #print('0',filename,numbers,numbers[0],min_ep)
                #print('match name',match_name)
                if(isNotEmpty(match_name)):
                    if(match_name in filename and int(numbers[0])>min_ep):
                        #print('1')
                        chkpt_max=os.path.join(dirName,filename)
                        min_ep=int(numbers[0])
                elif(int(numbers[0])>min_ep):
                    #print('2')
                    chkpt_max=os.path.join(dirName,filename)
                    min_ep=int(numbers[0])
    #print(chkpt_max)
    fin_chkpt_max = re.sub('\.meta$', '', chkpt_max)
    #print(fin_chkpt)
    return fin_chkpt_max

# Custom generator for mixup data
def mixup_data_gen(x_train,y_train,alpha=0.1):
    len_x_train = x_train.shape[0]
    x_out=np.zeros_like(x_train)
    y_out=np.zeros_like(y_train)

    for i in range(len_x_train):
        lam = np.random.beta(alpha, alpha)
        rand_idx1 = np.random.choice(len_x_train)
        rand_idx2 = np.random.choice(len_x_train)
        #print('i,lam',i,lam,1-lam,rand_idx1,rand_idx2)
        x_out[i] = lam * x_train[rand_idx1] + (1 - lam) * x_train[rand_idx2]
        y_out[i] = lam * y_train[rand_idx1] + (1 - lam) * y_train[rand_idx2]

    return x_out, y_out

def calculate_reg_target(masks, labels=4):
    targets = []
    for i in range(masks.shape[0]):
        areas = get_area(masks[i, :, :], labels=labels)
        target = areas/sum(areas)
    return np.asarray(target)

def get_area(mask, labels=4):
    nr_voxels = np.zeros((labels, 1))
    for i in range(labels):
        nr_voxels[i] = mask[mask==i].shape[0]
    return nr_voxels
