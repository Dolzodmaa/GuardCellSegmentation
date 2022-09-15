import numpy as np
import os
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
from tifffile import imread, imsave
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils import resize_to_512, make_patches

def denoise(img_path):

    train_dir = os.listdir(img_path)
    for image_name in train_dir:
 
        image = io.imread(img_path+image_name)
        sigma_est = np.mean(estimate_sigma(image,multichannel=False))
        nlm = denoise_nl_means(image, h=1.15*sigma_est, patch_size=5, patch_distance=3)
        nlm_uint = nlm.astype('uint16')
        imsave(img_path+image_name, nlm_uint)


def dataset_loader(img_dir, mask_dir, patch_shape, step):

    train_img_dir = os.listdir(img_dir)

    for i, img in enumerate(sorted(train_img_dir)):
        
        print(img)
        image = imread(img_dir+img)
        image = resize_to_512(image)
        image = (image - image.min())/(image.max()-image.min())
        img_patches = make_patches(image, patch_shape, step) 
        print(img_patches.shape)
        if i==0:
            input_img = img_patches
        else:
            input_img = np.concatenate((input_img, img_patches), axis = 0)
    
        mask = io.imread(os.path.join(mask_dir+'label-'+img))
        mask = resize_to_512(mask)
        mask = (mask - mask.min())/(mask.max()-mask.min())
        mask_patches = make_patches(mask, patch_shape, step)
        print('mask_shape', mask_patches.shape)
        if i==0:
            input_mask = mask_patches
        else:
            input_mask = np.concatenate((input_mask, mask_patches), axis = 0) 
      

    input_img = np.array(input_img).astype(np.float32)
    input_mask = np.array(input_mask).astype(np.float32)

    train_img = np.stack((input_img,)*3, axis=-1)
    train_mask = np.expand_dims(input_mask, axis=-1)

    del input_img
    del input_mask

    mask = []
    img = []
    for i in range(train_mask.shape[0]):
      if not train_mask[i, :, : , :, :].max()==0:
        mask.append(train_mask[i, :, : , :, :])
        img.append(train_img[i, :, : , :, :])

    train_img = np.reshape(img, (-1, 32, patch_shape, patch_shape, 3))
    train_mask = np.reshape(mask, (-1, 32, patch_shape, patch_shape, 1))

    del img
    del mask
    print(train_img.shape)
    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask, test_size = 0.20, random_state = 0, shuffle=False)
    
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)

    del train_img
    del train_mask
    
    return X_train, X_test, y_train, y_test
