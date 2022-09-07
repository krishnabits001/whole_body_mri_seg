__author__ = 'krishnch'

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import nibabel as nib

#to make directories
import pathlib
from skimage import transform

from  utils import *

import scipy.ndimage
from scipy.ndimage import morphology

class f1_utilsObj:
    def __init__(self,cfg,dt):
        print('f1 utils init')
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.batch_size=cfg.batch_size
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels
        self.interp_val = cfg.interp_val
        self.target_resolution=cfg.target_resolution
        self.data_path_tr=cfg.data_path_tr
        self.dt=dt

    def calc_pred_sf_mask_full(self, sess, ae, labeled_data_imgs, batch_factor=40):
        total_slices = labeled_data_imgs.shape[0]

        test_data = labeled_data_imgs

        #remainder
        rem=total_slices%batch_factor
        #quotient
        quo=int(total_slices/batch_factor)

        #print('rem,quo',rem,quo)

        for i in range(0,quo+1):
            if(i!=quo):
                no_of_slices=batch_factor
                test_data_tmp=np.reshape(test_data[i*batch_factor:(i+1)*batch_factor],(no_of_slices, self.img_size_x, self.img_size_y, 1))
            else:
                no_of_slices=rem
                test_data_tmp=np.reshape(test_data[i*batch_factor:(i*batch_factor+rem)],(no_of_slices, self.img_size_x, self.img_size_y, 1))

            seg_pred_tmp = sess.run(ae['y_pred'], feed_dict={ae['x']: test_data_tmp, ae['train_phase']: False})

            if(i==0):
                seg_pred=seg_pred_tmp
            else:
                seg_pred=np.concatenate((seg_pred,seg_pred_tmp),axis=0)

            #print('sp',seg_pred.shape,seg_pred_tmp.shape)

        return seg_pred
    
    def calc_pred_sf_mask_full_2channel(self, sess, ae, labeled_data_img1, labeled_data_img2, batch_factor=40):
        total_slices = labeled_data_img1.shape[0]

        test_data1 = labeled_data_img1
        test_data2 = labeled_data_img2

        #remainder
        rem=total_slices%batch_factor
        #quotient
        quo=int(total_slices/batch_factor)

        #print('rem,quo',rem,quo)

        for i in range(0,quo+1):
            if(i!=quo):
                no_of_slices=batch_factor
                test_fat_data_tmp=np.reshape(test_data1[i*batch_factor:(i+1)*batch_factor],(no_of_slices,self.img_size_x,self.img_size_y,1))
                test_wat_data_tmp=np.reshape(test_data2[i*batch_factor:(i+1)*batch_factor],(no_of_slices,self.img_size_x,self.img_size_y,1))
            else:
                no_of_slices=rem
                test_fat_data_tmp=np.reshape(test_data1[i*batch_factor:(i*batch_factor+rem)],(no_of_slices,self.img_size_x,self.img_size_y, 1))
                test_wat_data_tmp=np.reshape(test_data2[i*batch_factor:(i*batch_factor+rem)],(no_of_slices,self.img_size_x,self.img_size_y, 1))
            
            test_data_tmp=np.concatenate((test_fat_data_tmp,test_wat_data_tmp),axis=-1)

            seg_pred_tmp = sess.run(ae['y_pred'], feed_dict={ae['x']: test_data_tmp, ae['train_phase']: False})

            if(i==0):
                seg_pred=seg_pred_tmp
            else:
                seg_pred=np.concatenate((seg_pred,seg_pred_tmp),axis=0)

            #print('sp',seg_pred.shape,seg_pred_tmp.shape)

        return seg_pred 
    def reshape_img_and_f1_score(self, predicted_img_arr, mask, pixel_size):
        """
        :param predicted_img_arr:
        :param mask:
        :param pixel_size:
        :param target_resolution:
        :return:
        """
        #x, y = mask.shape[0],mask.shape[1]
        #x, y = 304,307
        nx,ny= self.img_size_x,self.img_size_y
        #print("nx,ny",nx,ny)

        scale_vector = (pixel_size[0] / self.target_resolution[0], pixel_size[1] / self.target_resolution[1])
        #predictions_arr = []
        mask_rescaled = transform.rescale(mask[:, :, 0], scale_vector, order=0, preserve_range=True, mode='constant')
        x, y = mask_rescaled.shape[0],mask_rescaled.shape[1]
        #print("2 reshape,x,y",mask_rescaled.shape,x,y)

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        total_slices = predicted_img_arr.shape[0]
        predictions_arr = np.zeros((mask.shape[0],mask.shape[1],total_slices))
        #print("t slices",total_slices)
        for slice_no in range(total_slices):
            # ASSEMBLE BACK THE SLICES
            slice_predictions = np.zeros((x,y,self.num_classes))
            predicted_img=predicted_img_arr[slice_no,:,:,:]
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = predicted_img
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, y_s:y_s+ny,:] = predicted_img[x_c:x_c+ x, :,:]
                elif x > nx and y <= ny:
                    slice_predictions[x_s:x_s + nx, :,:] = predicted_img[:, y_c:y_c + y,:]
                else:
                    slice_predictions[:, :,:] = predicted_img[x_c:x_c+ x, y_c:y_c + y,:]

            # RESCALING ON THE LOGITS
            prediction = transform.resize(slice_predictions,
                                              (mask.shape[0], mask.shape[1], self.num_classes),
                                              order=1,
                                              preserve_range=True,
                                              mode='constant')
            #print("b",prediction.shape)
            prediction = np.uint16(np.argmax(prediction, axis=-1))

            predictions_arr[:,:,slice_no]=prediction
            #print("a",prediction.shape,predictions_arr.shape)


        #print("pred shape",predictions_arr.shape,mask.shape)

        #Calculate F1 score
        y_pred= predictions_arr.flatten()
        y_true= mask.flatten()

        f1_val= f1_score(y_true, y_pred, average=None)
        #print("f1_val: {}".format(f1_val))
        return predictions_arr,f1_val

    def calc_f1_score(self,predicted_mask,gt_mask):
        #Calculate F1 score
        predicted_mask_tmp=np.argmax(predicted_mask,axis=-1)
        y_pred= predicted_mask_tmp.flatten()
        y_true= gt_mask.flatten()

        f1_val= f1_score(y_true, y_pred, average=None)

        return f1_val


    def pred_segs_bodyfat_test_subjs(self, sess,ae, save_dir,orig_img_dt,test_list,struct_name,only_segnet=0):

        mean_f1_arr=[]
        count=0

        for test_id in test_list:
            test_id_l=[test_id]

            #img_sys,label_sys= orig_img_dt(test_id_l,ret_affine=1)
            img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
            cropped_img_sys,cropped_mask_sys = self.dt.preprocess_data(img_sys, label_sys, pixel_size)

            crop_img_re,crop_labels_re= change_axis_img([cropped_img_sys,cropped_mask_sys])
            print("test data shape",img_sys.shape,label_sys.shape,cropped_img_sys.shape,cropped_mask_sys.shape)
            #print("crop test data shape",crop_img_re.shape,crop_labels_re.shape)

            # Make directory for that image
            seg_model_dir=str(save_dir)+'pred_segs/'+str(test_id)+'/'
            pathlib.Path(seg_model_dir).mkdir(parents=True, exist_ok=True)

            # Calc dice score and predicted segmentation & store in a txt file
            #pred_sf_mask = self.calc_pred_sf_mask_full(sess, ae, cropped_img_sys)
            #re_pred_mask_sys,f1_val_es = self.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)
            pred_sf_mask = self.calc_pred_sf_mask_full(sess, ae, crop_img_re)
            #print("0 pred data shape",crop_labels_re.shape,cropped_mask_sys.shape,pred_sf_mask.shape)
            re_pred_mask_sys,f1_val_es = self.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)
            #f1_val_es = calc_f1_score(re_pred_mask_sys,label_sys)
            #print("pred data shape",label_sys.shape,cropped_mask_sys.shape,crop_labels_re.shape,re_pred_mask_sys.shape)
            print("mean f1_val", f1_val_es)
            savefile_name = str(seg_model_dir)+'mean_f1_dice_coeff_test_id_'+str(test_id)+'.txt'
            np.savetxt(savefile_name, f1_val_es, fmt='%s')


            self.plot_predicted_seg_fat(img_sys,label_sys,re_pred_mask_sys,seg_model_dir,test_id)

            array_img = nib.Nifti1Image(re_pred_mask_sys.astype(np.int16), affine_tst)
            pred_filename = str(seg_model_dir)+'pred_seg_id_'+str(test_id)+'.nii.gz'
            nib.save(array_img, pred_filename)

            #f1_val_es=f1_val_es[1:self.num_classes]

            dsc_tmp_es=np.reshape(f1_val_es[1:self.num_classes], (1, self.num_classes - 1))
            #mean_f1_arr.append(np.mean(dsc_tmp_es))

            if(count==0):
                dsc_all_es=dsc_tmp_es
                count=1
            else:
                dsc_all_es=np.concatenate((dsc_all_es, dsc_tmp_es))

        #for DSC
        val_list=[]
        val_list_mean=[]
        for i in range(0,self.num_classes-1):
            dsc=dsc_all_es[:,i]
            #print("ES dsc_vae", dsc)
            #print(round(np.mean(dsc), 3), ',', round(np.median(dsc), 3), ',', round(np.std(dsc), 3))
            #DSC
            val_list.append(round(np.mean(dsc), 3))
            val_list.append(round(np.median(dsc), 3))
            val_list.append(round(np.std(dsc), 3))
            val_list_mean.append(round(np.mean(dsc), 3))
            filename_save=str(save_dir)+'pred_segs/'+str(struct_name[i])+'_20subjs_dsc.txt'
            np.savetxt(filename_save,dsc,fmt='%s')
        filename_save=str(save_dir)+'pred_segs/'+'mean_median_std_dsc.txt'
        np.savetxt(filename_save,val_list,fmt='%s')
        filename_save=str(save_dir)+'pred_segs/'+'dsc_mean.txt'
        np.savetxt(filename_save,val_list_mean,fmt='%s')
        filename_save=str(save_dir)+'pred_segs/'+'net_dsc_mean.txt'
        net_mean_dsc=[]
        net_mean_dsc.append(round(np.mean(val_list_mean),3))
        np.savetxt(filename_save,net_mean_dsc,fmt='%s')



    def plot_predicted_seg_fat(self, test_data_img,test_data_labels,predicted_labels,save_dir,test_id):
        #n_examples=3
        #fig, axs = plt.subplots(4, 3, figsize=(12, 10))
        #range_list=[100,150,200,250,300]
        max_val = test_data_img.shape[2]
        n_imgs=4
        #print('v',n_imgs,max_val)
        #print('v',test_data_img.shape,test_data_labels.shape,predicted_labels.shape)
        if(max_val<201):
            min_val=max_val
        else:
            min_val=201
        j=0
        plt.figure(figsize=(16,12))
        plt.suptitle('Predicted Seg',fontsize=10)
        for example_i in range(50,min_val,50):
        #for example_i in range(50,201,50):
            #print('j',j,example_i)
            plt.subplot(n_imgs,3,3*j+1)
            if(j==0):
                plt.title('Test Img')
            plt.imshow(test_data_img[:,:,example_i],cmap='gray')
            plt.axis('off')
            plt.subplot(n_imgs,3,3*j+2)
            if(j==0):
                plt.title('GT label')
            plt.imshow(test_data_labels[:,:,example_i])#,cmap='gray')
            plt.axis('off')
            plt.subplot(n_imgs,3,3*j+3)
            if(j==0):
                plt.title('Pred label')
            plt.imshow(np.squeeze(predicted_labels[:,:,example_i]))#,cmap='gray')
            plt.axis('off')

            j=j+1
        savefile_name=str(save_dir)+'tst'+str(test_id)+'_pred_seg_imgs_p1.png'
        plt.savefig(savefile_name)
        plt.close('all')

        n_imgs=int(max_val/50)-4
        j=0
        plt.figure(figsize=(20,12))
        plt.suptitle('Predicted Seg',fontsize=10)
        for example_i in range(250,max_val,50):
            #print('j',j,example_i)
            plt.subplot(n_imgs,3,3*j+1)
            if(j==0):
                plt.title('Test Img')
            plt.imshow(test_data_img[:,:,example_i],cmap='gray')
            plt.axis('off')
            plt.subplot(n_imgs,3,3*j+2)
            if(j==0):
                plt.title('GT label')
            plt.imshow(test_data_labels[:,:,example_i])#,cmap='gray')
            plt.axis('off')
            plt.subplot(n_imgs,3,3*j+3)
            if(j==0):
                plt.title('Pred label')
            plt.imshow(np.squeeze(predicted_labels[:,:,example_i]))#,cmap='gray')
            plt.axis('off')

            j=j+1
        savefile_name=str(save_dir)+'tst'+str(test_id)+'_pred_seg_imgs_p2.png'
        plt.savefig(savefile_name)
        plt.close('all')

    def surfd(self,input1, input2, sampling=1, connectivity=1):
        '''
        #for computing surface distance
        :param input1: predicted labels
        :param input2: ground truth
        :param sampling: default value
        :param connectivity: default value
        :return: sds : surface distance
        '''
        input_1 = np.atleast_1d(input1.astype(np.bool))
        input_2 = np.atleast_1d(input2.astype(np.bool))
        #print(input_1.dtype,input_1.shape,input_1.ndim)
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
        #print(conn)

        #S = input_1 - morphology.binary_erosion(input_1, conn)
        y=morphology.binary_erosion(input_1, conn)
        y=y.astype(np.float32)
        x=input_1.astype(np.float32)
        S=x-y
        #print(z)
        #print(x.shape,x.dtype)

        #Sprime = input_2 - morphology.binary_erosion(input_2, conn)
        y=morphology.binary_erosion(input_2, conn)
        y=y.astype(np.float32)
        x=input_2.astype(np.float32)
        Sprime=x-y

        S=S.astype(np.bool)
        Sprime=Sprime.astype(np.bool)

        dta = morphology.distance_transform_edt(~S,sampling)
        dtb = morphology.distance_transform_edt(~Sprime,sampling)

        sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

        return sds

    def assign_1hot_enc(self,batch_labels):
        len_of_batch=batch_labels.shape[0]
        batch_labels_1hot=np.zeros((len_of_batch,self.img_size_x,self.img_size_y,self.num_classes))
        #print('1',batch_labels_1hot.shape,batch_labels.shape)
        for index in range(0,len_of_batch):
            batch_tmp_label_test=batch_labels[index,:,:]
            for i in range(0,self.num_classes):
                tmp_arr=np.zeros((self.img_size_x,self.img_size_y))
                #print('2',batch_tmp_label_test.shape,tmp_arr.shape)
                tmp_arr[batch_tmp_label_test==i]=1
                batch_labels_1hot[index,:,:,i]=tmp_arr
        return batch_labels_1hot

    #def write_gif_func(self, y_pred, imsize, save_dir):
    #    #y_pred.shape
    #    y = np.squeeze(y_pred)
    #    y_t=np.transpose(y)
    #    #y_f = np.reshape(y_t,(320*320,20))
    #    recons_ims = np.reshape(y_t,(self.img_size_x*self.img_size_y,self.batch_size))
    #    #y_f.shape

    #    dataset =np.transpose(recons_ims.reshape(1,imsize[0],imsize[1],recons_ims.shape[1]),[3,0,1,2])
    #    np.expand_dims(dataset, axis=1)
    #    dataset = np.tile(dataset, [1,3,1,1])
    #    imname=save_dir+'plots/test_slice.gif'
    #    write_gif((dataset*256).astype(np.uint8), imname, fps=5)
