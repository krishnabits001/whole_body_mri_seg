#!/bin/bash
ip_path='/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/bodyfat_orig/top_down_nifti'
for pid in {1..70}
do
   for timept in 1 2
   do 
      echo "patient id $pid, time point $timept"
      FILE=$ip_path'/top_down_niftyp'$pid'_z'$timept'_f.nii.gz'
      #echo $FILE
      if [ -f "$FILE" ]; then
         echo "$FILE exists. Running Inference on it."
         python inference_mri.py --ip_path_fat=$FILE --patient_id=$pid --time_pt=$timept
         #give out_path here to save the predicted masks for all subjects or define it once inside inference_mri.py file.
         #python inference_mri.py --ip_path_fat=$FILE --out_path=<directory to save all predicted masks> --patient_id=$pid --time_pt=$timept
      fi
   done
done
