import numpy as np
import skimage
from skimage.filters import gaussian

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



