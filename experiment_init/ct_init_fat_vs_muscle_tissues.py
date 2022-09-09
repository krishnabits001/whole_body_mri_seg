################################################################
# Definitions required for CNN graph
################################################################
#Interpolation type and up scaling factor
interp_val=0 # 0 - bilinear interpolation; 1- nearest neighbour interpolation
################################################################

################################################################
# data dimensions, num of classes and resolution
################################################################
#Data Dimensions
img_size_x = 320
img_size_y = 224
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size_x * img_size_y
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Number of classes : # 0-background, 1-abdominal muscle, 2-paravertebral muscle, 3-psoas muscle, 4-subcutaneous fat, 5-visceral fat
num_classes=6
size=(img_size_x,img_size_y)
target_resolution=(1.5,1.5)
################################################################
#data paths
################################################################
#validation_update_step to save values
val_step_update=20
#base dir of network
#base_dir='/scratch_net/flup/fsamuel/bodyfat_seg/'
#base_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/whole_body_mri_seg/'
base_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/'
#data path tr
data_path_tr='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/new_scan_data/ct/'
#cropped imgs data_path
data_path_tr_cropped='/usr/bmicnas01/data-biwi-01/krishnch/projects/bodyfat_seg/fin_git_repo/new_scan_data/ct/'
################################################################

################################################################
#network optimization parameters
################################################################
#use dice score foreground (1) or weighted cross entropy (0) for loss function optimizer
dsc_l_val=1
#enable data augmentation
aug_en=1
#learning rate for segmentation net
lr=0.001
#learning rate for reconstruction task
lr_re=0.001
#batch_size
batch_size=20
struct_name=['abdomin_musc','parav_musc','psoas_musc','sub_fat','visc_fat']
