import numpy as np
import skimage
from skimage.filters import gaussian
import os
import glob

IMAGE_DIR = '/home/nishchal/workspace/MIT/6861/ILSVRC2013_DET_val/done'
MASK_DIR = '/home/nishchal/workspace/MIT/6861/saliency-salgan-2017/imagenet_saliency'
IMAGE_NAMES = []
for img_name in glob.glob(os.path.join(MASK_DIR, '*')):
    IMAGE_NAMES.append(os.path.split(img_name)[-1])

def make_masked_image(image, mask_img, threshold, blur=False, blur_amount=10):
    mask = mask_img <= threshold
    result = image.copy()
    if blur:
        back_image = gaussian(image, blur_amount)
        back_image = skimage.img_as_ubyte(back_image)
    else:
        back_image = np.zeros(image.shape)
    result[mask] = back_image[mask]
    return result

def test_image_at_levels(image_name, level_list, blur=False, blur_amount=10):
    img = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
    mask_img = skimage.io.imread(os.path.join(MASK_DIR, image_name))
    for level in level_list:
        masked_image = make_masked_image(img, mask_img, level, blur, blur_amount)
        # TODO: VGG classify stuff, store results
