This code is for whole body segmentation of MRIs for the following labels: subcutaneous fat, visceral fat and muscle.

Requirements:</br>
Python 3.6.1,</br>
Tensorflow 1.12.0,</br>
rest of the requirements are mentioned in the "requirements.txt" file.

I) To clone the git repository.</br>
git clone <repo_name>.git </br>

II) Install python, required packages and tensorflow.</br>
Then, install python packages required using below command or the packages mentioned in the file.</br>
pip install -r requirements.txt </br>

To install tensorflow </br>
pip install tensorflow-gpu=1.12.0 </br>

III) To train the baseline with affine transformations for comparison, use the below code file. (check train_model/train_unet_script.sh) </br>
cd train_model/ </br>
python train_baseline_unet.py --dataset=diff_fat --no_of_tr_imgs=tr8 --comb_tr_imgs=c4 --data_aug=1 --dsc_loss=0 --ver=0
 </br>

IV) To infer the predicted mask for any new test image, use the below script (train_model/inference_script.sh) </br>
cd train_model/ </br>
python inference.py --ip_path=<path_of_input_image> --out_path=<path_to_save_predicted_segmentation_mask>
</br>

V) Config files contents. </br>
One can modify the contents of the below 2 config files to run the required experiments. </br>
experiment_init directory contains 2 files. </br>
   init_sub_vs_visc_fat.py </br>
   --> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models. Target resolution of (1.5,1.5) and image dimensions of (320,224) used for training of the model are mentioned in this file. </br>
   data_cfg_bodyfat.py </br>
   --> contains an example of data config details where one can set the patient ids which they want to use as train, validation and test images. </br>

