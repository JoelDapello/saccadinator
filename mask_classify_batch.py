
# coding: utf-8

# In[1]:

import numpy as np
import skimage
import skimage.io
from skimage.filters import gaussian
import os
import glob
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from RunMain import VGG_16
import CallResult
import matplotlib.pyplot as plt
import h5py as h5
import cv2


# In[33]:

def make_masked_image(image, mask_img, threshold, blur=False, blur_amount=10):
    mask = mask_img < threshold
    result = image.copy()
    if blur:
        # back_image = gaussian(image, blur_amount)
        # back_image = skimage.img_as_ubyte(back_image)
        back_image = cv2.GaussianBlur(image, (0,0), blur_amount)
    else:
        back_image = np.zeros(image.shape)
    result[mask] = back_image[mask]
    return result

def compile_results(out):
    ordered_idx = np.argsort(-out)
            
    full_result = [(
        CallResult.lines[int(idx)].split()[0], 
        CallResult.lines[int(idx)], 
        idx, 
        out[0][idx]) 
        for idx in ordered_idx[0]
    ]
    
    return full_result

def test_image_at_levels(image_name, percentages, model, blur=False, blur_amount=10):
    """ Perform the classification accuracy test for the given image

    Args:
        image_name (string): The name of the image to test (should be in IMAGE_DIR)
        percetages list(int): A list of percentiles to use for tests e.g. [25,50,100]
        model object: the model to use for classification
        blur (boolean): whether to blur or leave background black
        blur_amount (float): the amounf of blur to apply
    """
    results_dict = {
        'valid_WNID' : VAL_LABELS[get_im_idx(image_name)-1],
        'valid_callresult' : get_VGG_map(VAL_LABELS[get_im_idx(image_name)-1]),
        'image_name' : image_name,
        'percentages' : percentages,
        'full_results' : [],
        'trajectories' : {}
    }
    print(results_dict)
    # img = skimage.io.imread(os.path.join(IMAGE_DIR, image_name[:-3]+'JPEG'))
    img = cv2.imread(os.path.join(IMAGE_DIR, image_name[:-3]+'JPEG'))
#     plt.imshow(img)
    # mask_img = skimage.io.imread(os.path.join(MASK_DIR, image_name))
    mask_img = cv2.imread(os.path.join(MASK_DIR, image_name))
    results = []
    level_list = get_ntiles_for_img(mask_img, percentages)
#     print(level_list)
    for level in level_list:
        masked_image = make_masked_image(img, mask_img, level, blur, blur_amount)
#         cv2.imshow('img',masked_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        # Transform image for VGG
        masked_image = cv2.resize(masked_image, (224,224)).astype(np.float32)
        masked_image[:,:,0] -= 103.939
        masked_image[:,:,1] -= 116.779
        masked_image[:,:,2] -= 123.68
        masked_image = masked_image.transpose((1,0,2))
        masked_image = np.expand_dims(masked_image, axis=0)
        out = model.predict(masked_image)
        full_result = compile_results(out)
        
        results_dict['full_results'].append(full_result)
        if 'best_guess' not in results_dict: 
            results_dict['best_guesses'] = (full_result[0:5],level)
            
        # loop all softmax outputs
        for result in full_result:
            # make sure WNID [] exists
            if result[0] not in results_dict['trajectories']: results_dict['trajectories'][result[0]] = []
            results_dict['trajectories'][result[0]].append(result[-1])

    return results_dict

def get_ntiles_for_img(img, percentages):
    """ Split calculate """
    percentiles = []
    for i in percentages:
        percentiles.append(np.percentile(img, i))
    return percentiles

def get_im_idx(file):
    return int(file.split('_')[-1].split('.')[0])

def get_VGG_map(WNID):
    match = None
    for line in CallResult.lines:
        if WNID in line: match = line
    return match


# In[3]:

# Make VGG model
VGG_WEIGHTS = '/n/regal/cox_lab/dapello/VGG16_Keras_TensorFlow/data/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model = VGG_16(VGG_WEIGHTS)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


# In[4]:

IMAGE_DIR = '/n/regal/cox_lab/dapello/saliency-salgan-2017/images/ILSVRC2012_val/'
MASK_DIR = '/n/regal/cox_lab/dapello/saliency-salgan-2017/saliency/ILSVRC2012_val/'
IMAGE_NAMES = []
VAL_LABELS = [line.replace('\n','') for line in open('ground_truth_sane.txt')]

for img_name in glob.glob(os.path.join(MASK_DIR, '*')):
    IMAGE_NAMES.append(os.path.split(img_name)[-1])


# In[34]:

results_array = []
ntiles = np.arange(0,101,10)
for image_name in IMAGE_NAMES:
    results_dict = test_image_at_levels(image_name, ntiles, model, blur=True)
    results_array.append(results_dict)

results_array_np = np.array(results_array)
np.savez('results_array',results_array_np)
