import numpy as np
import os
import glob
import matplotlib.pyplot as plt


IMAGE_DIR = '/home/nishchal/workspace/MIT/6861/ILSVRC2013_DET_val/done'
MASK_DIR = '/home/nishchal/workspace/MIT/6861/saliency-salgan-2017/imagenet_saliency'
IMAGE_NAMES = []
VGG_WEIGHTS = '/home/nishchal/workspace/MIT/6861/saccadinator/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
for img_name in glob.glob(os.path.join(MASK_DIR, '*')):
    IMAGE_NAMES.append(os.path.split(img_name)[-1])

# Make VGG model
# model = VGG_16(VGG_WEIGHTS)
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')

def plot_accuracy_vs_saliency(results_array):
    correct = np.zeros(11)
    for r in results_array:
        for i in range(11):
            best_guess = sorted(r['trajectories'], reverse=True,key=lambda x: r['trajectories'][x][i])[0]
            print(best_guess, r['valid_WNID'])
            if best_guess == r['valid_WNID']:
                correct[i] += 1
    return correct/float(len(results_array))

def plot_confidence_vs_saliency(results_array_elem):
    return results_array_elem['trajectories'][results_array_elem['valid_WNID']]



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

def test_image_at_levels(image_name, percentages, blur=False, blur_amount=10):
    """ Perform the classification accuracy test for the given image

    Args:
        image_name (string): The name of the image to test (should be in IMAGE_DIR)
        percetages list(int): A list of percentiles to use for tests e.g. [25,50,100]
        blur (boolean): whether to blur or leave background black
        blur_amount (float): the amounf of blur to apply
    """
    # img = skimage.io.imread(os.path.join(IMAGE_DIR, image_name[:-3]+'JPEG'))
    img = cv2.imread(os.path.join(IMAGE_DIR, image_name[:-3]+'JPEG'))
    # mask_img = skimage.io.imread(os.path.join(MASK_DIR, image_name))
    mask_img = cv2.imread(os.path.join(MASK_DIR, image_name))
    results = []
    level_list = get_ntiles_for_img(mask_img, percentages)
    print(level_list)
    for level in level_list:
        masked_image = make_masked_image(img, mask_img, level, blur, blur_amount)
        cv2.imshow('img',masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Transform image for VGG
        masked_image = cv2.resize(masked_image, (224,224)).astype(np.float32)
        masked_image[:,:,0] -= 103.939
        masked_image[:,:,1] -= 116.779
        masked_image[:,:,2] -= 123.68
        masked_image = masked_image.transpose((1,0,2))
        masked_image = np.expand_dims(masked_image, axis=0)
        out = model.predict(masked_image)
        ordered_idx = np.argsort(-out)
        print(out.max(), ordered_idx[0][0])
        result = (CallResult.lines[int(ordered_idx[0][0])], out[0][ordered_idx[0]][0])
        results.append(result)

    return results

def get_ntiles_for_img(img, percentages):
    """ Split calculate """
    percentiles = []
    for i in percentages:
        percentiles.append(np.percentile(img, i))
    return percentiles
