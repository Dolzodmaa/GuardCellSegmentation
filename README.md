# Stomata Segmentation

Stomata segmentation is a tool for segmenting all guard cells in 3D volumetric images.
The pipeline uses a patch-wise, attention gated, 3D U-Net like architecture with weighted combination of Dice and Focal Loss functions as a segmentation tool. The resulting output from the segmentation model is then post-processed to refine the result and separate each instance of guard cell. 

## Install dependencies

 ```bash
 pip3 install -r requirements.txt
 ```
 
 ## Run a segmentation inference on a test image
 
  ```bash
 python3 predict.py --input 'test.tif' --model_dir '/log/' --model_name128 'p_128_model.epoch95-loss0.91.hdf5' --model_name256 'p_256_model.epoch28-loss0.69.hdf5'
 ```
 
 ## Train on a custom dataset
 
  ```bash
 python3 train.py --img_dir '/img_train/' --mask_dir '/img_label/' --log_dir '/log/' --patch_shape 256 --patch_step 128 --epochs 1000
 ```
 
