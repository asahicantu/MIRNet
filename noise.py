import numpy as np
import os
import cv2
import findpeaks

DENOISE_TYPE_CV_NLM = 'cv_nlm'
DENOISE_TYPE_SELF_NLM = 'self_nlm'
DENOISE_TYPE_MEDIAN = 'median'

def noisify_speckle(image):
    gauss = None
    if len(image.shape) == 2:
        row, col = image.shape
        mu, sigma = 0, 0.7
        # gauss = np.random.normal(mu, sigma, row * col).astype('uint8')
        gauss = np.random.randn(row, col).astype('uint8')
        gauss = gauss.reshape(row, col)
    elif len(image.shape) == 3:
        row, col, ch = image.shape
        # gauss = np.random.randn(mu, sigma, row * col * ch)
        gauss = np.random.randn(row, col, ch).astype('uint8')
        gauss = gauss.reshape(row, col, ch) 
    noisy = image + image * gauss
    return noisy

def non_local_means(image, big_window, small_window):
    if (big_window % 2 != 0 or small_window % 2 != 0):
        # Window size is not even so cannot pad image
        return
    
    padding_width = int(big_window / 2)
    # print(padding_width)
    new_image_x = int(image.shape[0] + (2 * padding_width))
    # print(new_image_x)
    new_image_y = int(image.shape[1] + (2 * padding_width))
    # print(new_image_x)
    
    # Padding the image with to be able to loop with big window
    padded_image = np.zeros(new_image_x * new_image_y).astype(np.uint8).reshape(new_image_x, new_image_y)
    padded_image[padding_width:new_image_x - padding_width, padding_width:new_image_y - padding_width] = image
    # Adding flipped image on the edges to reduce the effect of counting 0s in algorithm vertically
    # x_flipped_image = np.fliplr(image)
    # print(x_flipped_image)
    padded_image[padding_width:new_image_x - padding_width, 0:padding_width] = np.fliplr(image[:, 0:padding_width])
    padded_image[padding_width:new_image_x - padding_width, new_image_y - padding_width:new_image_y] = np.fliplr(image[:, new_image_y - (3 * padding_width):new_image_y - (2 * padding_width)])
    # y_flipped_image = np.flipud(image)
    # print(y_flipped_image)
    # Adding flipped image on the edges to reduce the effect of counting 0s in algorithm vertically
    padded_image[0:padding_width, :] = np.flipud(padded_image[padding_width:(2 * padding_width), :])
    padded_image[new_image_x - padding_width:new_image_x, :] = np.flipud(padded_image[new_image_x - (2 * padding_width):new_image_x - padding_width, :])
    result = padded_image.copy()
    small_half_width = int(small_window / 2)
    for i in range(padding_width, new_image_y - padding_width):
        for j in range(padding_width, new_image_x - padding_width):

            big_win_x = i - padding_width
            big_win_y = j - padding_width
            region = padded_image[j - small_half_width:j + small_half_width + 1, i - small_half_width:i + small_half_width + 1]
            pixel_color = 0
            total_weight = 0
            for small_win_x in range(big_win_x, big_win_x + big_window - small_window, 1):
                for small_win_y in range(big_win_y, big_win_y + big_window - small_window, 1):   
                    small_region = padded_image[small_win_y:small_win_y + small_window + 1, small_win_x:small_win_x + small_window + 1]
                    euclid_dist = np.sqrt(np.sum(np.square(small_region - region)))
                    weight = np.exp(-euclid_dist / 16)
                    total_weight += weight
                    pixel_color += weight * padded_image[small_win_y + small_half_width, small_win_x + small_half_width]

            pixel_color /= total_weight
            result[j, i] = pixel_color

    return result[padding_width:new_image_x - (2 * padding_width), padding_width:new_image_y - (2 * padding_width)]

def denosify(denoise_type, img, params = tuple()):
    denoise = None
    if denoise_type == DENOISE_TYPE_CV_NLM:
        denoise = cv2.fastNlMeansDenoising(img, None)
    if denoise_type == DENOISE_TYPE_SELF_NLM:
        big_window, small_window = params
        denoise = non_local_means(img, big_window, small_window)
    if denoise_type == DENOISE_TYPE_MEDIAN:
        window = params
        denoise = findpeaks.median_filter(img, win_size = window) 
    return denoise


def get_histogram(img, idx):
    histr = cv2.calcHist([img],[idx],None,[256],[0,256])
    return histr