__author__ = 'krishnch'

import numpy as np
import nibabel as nib
from skimage import transform

class dataloaderObj:

    #define functions to load data from different organs
    def __init__(self,cfg):
        print('dataloaders init')
        self.data_path_tr=cfg.data_path_tr
        self.data_path_tr_cropped=cfg.data_path_tr_cropped
        self.target_resolution=cfg.target_resolution
        self.size=cfg.size
        self.num_classes=cfg.num_classes
        print('nc',self.num_classes)

    def normalize_minmax_data(self, image_data):
        """
        3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
        Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
        We expect the outliers from noise and scanner calibrations to be away from the range of [0,1].
        Args:
            image_data : 3D MRI scan to be normalized using min-max normalization
        Returns:
            final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
        """
        min_val_2p=np.percentile(image_data,1)
        max_val_98p=np.percentile(image_data,99)
        #print("min,max val",min_val_2p,max_val_98p)
        final_image_data=np.zeros((image_data.shape[0],image_data.shape[1],image_data.shape[2]), dtype=np.float64)
        #Total volume
        final_image_data=(image_data-min_val_2p)/(max_val_98p-min_val_2p)
        return final_image_data

    def load_diff_fat_types_img_labels(self, train_ids_list,ret_affine=0,label_present=1):
        #Load the body fat data and its labels
        for study_id in train_ids_list:
            #print("diff study_id",study_id)
            img_fname = str(self.data_path_tr)+str(study_id)+'/fat_img.nii.gz'
            img_load=nib.load(img_fname)
            img_tmp=img_load.get_data()
            pixel_size=img_load.header['pixdim'][1:4]
            affine_tst=img_load.affine
            if(label_present==1):
                mask_fname = str(self.data_path_tr)+str(study_id)+'/mask.nii.gz'
                mask_load=nib.load(mask_fname)
                label_tmp=mask_load.get_data()

        #print('before norm',np.min(img_tmp),np.max(img_tmp),np.mean(img_tmp))
        #normalize each 3D image separately
        img_tmp=self.normalize_minmax_data(img_tmp)
        #print('after norm',np.min(img_tmp),np.max(img_tmp),np.mean(img_tmp))
        if(label_present==1):
            label_re=np.copy(label_tmp)
            #label_re[label_re==1]=0
            label_re[label_tmp==12]=1
            label_re[label_tmp==13]=2
            label_re[label_tmp==14]=2
            #add label for muscle to 3
            # Trapezius 
            label_re[label_tmp==1]=3
            # Deltoideus
            label_re[label_tmp==2]=3
            # Infraspinatus
            label_re[label_tmp==3]=3
            # Iliacus
            label_re[label_tmp==4]=3
            # Psoas
            label_re[label_tmp==5]=3
            # Gluteus maximus
            label_re[label_tmp==7]=3
            # Gluteus medius
            label_re[label_tmp==8]=3
            # Gluteus minimus
            label_re[label_tmp==9]=3
            # Quadriceps
            label_re[label_tmp==10]=3
            # Ischicrural
            label_re[label_tmp==11]=3
            # Supraspinatus
            label_re[label_tmp==19]=3
            # Subscapularis
            label_re[label_tmp==20]=3
            # Teres minor
            label_re[label_tmp==21]=3
                # paraspinal muscles - not found in 1_seg
                # lower leg posterior - not found in 1_seg
                # lower leg anterior - not found in itksnap
            if(study_id=='004' or study_id=='005' or study_id=='018' or study_id=='019' or study_id=='020' or study_id=='021' or\
               study_id=='023' or study_id=='025' or study_id=='027' or study_id=='028' or study_id=='038' or study_id=='045'):
                # Adductors
                label_re[label_tmp==22]=3
                # Rueckenmuskulatur
                label_re[label_tmp==23]=3
                # Obturatorius
                label_re[label_tmp==27]=3
                # Tensor fasciae latae
                label_re[label_tmp==28]=3
                # compartimentum posterios + anterior
                label_re[label_tmp==29]=3
                label_re[label_tmp==30]=3
                # other muscle
                label_re[label_tmp==31]=3
                # Fibula 
                label_re[label_tmp==24]=0
                # Tibia
                label_re[label_tmp==25]=0
                # Patella
                label_re[label_tmp==26]=0
            elif(study_id=='006' or study_id=='007' or study_id=='010' or study_id=='011'):
                # Adductors
                label_re[label_tmp==22]=3
                # Rueckenmuskulatur
                label_re[label_tmp==23]=3
                # other muscle
                label_re[label_tmp==24]=3
            elif(study_id=='013'):
                # Adductors
                label_re[label_tmp==22]=3
                # Rueckenmuskulatur
                label_re[label_tmp==23]=3
                # other muscle
                #label_re[label_tmp==24 or label_tmp==25 or label_tmp==26 or label_tmp==27 or label_tmp==28 or label_tmp==29 or label_tmp==30]=3
            elif(study_id=='015'):
                # Adductors
                label_re[label_tmp==25]=3
                # Rueckenmuskulatur
                label_re[label_tmp==22]=3
                # other muscle
                label_re[label_tmp==23]=3
                label_re[label_tmp==24]=3
                label_re[label_tmp==26]=3
                label_re[label_tmp==27]=3
                label_re[label_tmp==28]=3
                label_re[label_tmp==29]=3
                label_re[label_tmp==30]=3
                label_re[label_tmp==31]=3
            elif(study_id=='016'):
                # Adductors
                label_re[label_tmp==22]=3
                # Rueckenmuskulatur
                label_re[label_tmp==23]=3
                # other muscle
                label_re[label_tmp==24]=3
                label_re[label_tmp==25]=3
                label_re[label_tmp==26]=3
                label_re[label_tmp==27]=3
                label_re[label_tmp==28]=3
                label_re[label_tmp==29]=3
                label_re[label_tmp==30]=3
            elif(study_id=='001'):
                # Adductors
                label_re[label_tmp==25]=3
                # Rueckenmuskulatur
                label_re[label_tmp==22]=3
                # Obturatorius
                label_re[label_tmp==29]=3
                # other muscle
                label_re[label_tmp==23]=3
                label_re[label_tmp==24]=3
                label_re[label_tmp==30]=3
                label_re[label_tmp==31]=3
                label_re[label_tmp==32]=3
                label_re[label_tmp==33]=3
                # Fibula 
                label_re[label_tmp==25]=0
                # Tibia
                label_re[label_tmp==27]=0
                # Patella
                label_re[label_tmp==28]=0
            # Femur
            label_re[label_tmp==18]=0
            # set rest to background
            label_re[label_re>3]=0
        print('np.unique',np.unique(label_re))

        if(label_present==0):
            return img_tmp,pixel_size
        else:
            if(ret_affine==0):
                return img_tmp,label_re,pixel_size
            else:
                return img_tmp,label_re,pixel_size,affine_tst

    def load_fat_img_labels(self, train_ids_list,ret_affine=0,label_present=1):
        #Load the body fat data and its labels
        for study_id in train_ids_list:
            #print("study_id",study_id)
            img_fname = str(self.data_path_tr)+str(study_id)+'/fat_img.nii.gz'
            img_load=nib.load(img_fname)
            img_tmp=img_load.get_data()
            pixel_size=img_load.header['pixdim'][1:4]
            affine_tst=img_load.affine
            if(label_present==1):
                mask_fname = str(self.data_path_tr)+str(study_id)+'/mask.nii.gz'
                mask_load=nib.load(mask_fname)
                label_tmp=mask_load.get_data()

        #print('before norm',np.min(img_tmp),np.max(img_tmp),np.mean(img_tmp))
        #normalize each 3D image separately
        img_tmp=self.normalize_minmax_data(img_tmp)
        #print('after norm',np.min(img_tmp),np.max(img_tmp),np.mean(img_tmp))
        if(label_present==1):
            label_re=np.copy(label_tmp)
            label_re[label_re==1]=0
            label_re[label_re==12]=1
            label_re[label_re==13]=1
            label_re[label_re==14]=1
            label_re[label_re!=1]=0

        if(label_present==0):
            return img_tmp,pixel_size
        else:
            if(ret_affine==0):
                return img_tmp,label_re,pixel_size
            else:
                return img_tmp,label_re,pixel_size,affine_tst

    def crop_or_pad_slice_to_size(self, img_slice, nx, ny):

        slice_cropped=np.zeros((nx,ny))
        x, y = img_slice.shape

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        if x > nx and y > ny:
            slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

        return slice_cropped


    def preprocess_data(self, img, mask, pixel_size,label_present=1):

        nx,ny=self.size

        scale_vector = [pixel_size[0] / self.target_resolution[0],
                        pixel_size[1] / self.target_resolution[1]]

        #cropped_img=np.zeros_like(img)
        #cropped_mask=np.zeros_like(mask)

        for slice_no in range(img.shape[2]):

            slice_img = np.squeeze(img[:, :, slice_no])
            slice_rescaled = transform.rescale(slice_img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               mode = 'constant')
            if(label_present==1):
                slice_mask = np.squeeze(mask[:, :, slice_no])
                mask_rescaled = transform.rescale(slice_mask,
                                              scale_vector,
                                              order=0,
                                              preserve_range=True,
                                              mode='constant')

            slice_cropped = self.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
            if(label_present==1):
                mask_cropped = self.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)
            #print(slice_rescaled.shape,mask_rescaled.shape)

            if(slice_no==0):
                cropped_img=np.reshape(slice_cropped,(nx,ny,1))
                if(label_present==1):
                    cropped_mask=np.reshape(mask_cropped,(nx,ny,1))
            else:
                slice_cropped_tmp=np.reshape(slice_cropped,(nx,ny,1))
                cropped_img=np.concatenate((cropped_img,slice_cropped_tmp),axis=2)
                if(label_present==1):
                     mask_cropped_tmp=np.reshape(mask_cropped,(nx,ny,1))
                     cropped_mask=np.concatenate((cropped_mask,mask_cropped_tmp),axis=2)

            #print(slice_cropped.shape,cropped_img.shape)
            #print(mask_cropped.shape,cropped_mask.shape)
        if(label_present==1):
            return cropped_img,cropped_mask
        else:
            return cropped_img

    def load_cropped_img_labels(self, train_ids_list,label_present=1):
    #Load the cropped data and its labels
        count=0
        for study_id in train_ids_list:
            #print("study_id",study_id)
            img_fname = str(self.data_path_tr_cropped)+str(study_id)+'/img_cropped.npy'
            img_tmp=np.load(img_fname)
            if(label_present==1):
                mask_fname = str(self.data_path_tr_cropped)+str(study_id)+'/mask_cropped.npy'
                mask_tmp=np.load(mask_fname)
                # if(self.one_label==1):
                #     print('only 1 label')
                #     mask_tmp[mask_tmp==2]=1
            if(count==0):
                img_cat=img_tmp
                if(label_present==1):
                    mask_cat=mask_tmp
                count=1
            else:
                img_cat=np.concatenate((img_cat,img_tmp),axis=2)
                if(label_present==1):
                    mask_cat=np.concatenate((mask_cat,mask_tmp),axis=2)
            #print(img_tmp.shape,img_cat.shape)
        if(label_present==1):
            return img_cat,mask_cat
        else:
            return img_cat
