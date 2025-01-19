import numpy as np

def gamma_correct(image, gamma):
    """ Function to adjust contrast of the image
    """
    return np.power(image, gamma)


def adjust_brightness(image, gain_array):
    """ Function to adjust brightness of the image
        Number of elements in the gain_array should match the of channels of the image 
    """
    return image * gain_array


def normalise_pixels(image):
    """ Function to normalise the pixels
        Step 1: Calculate minimum and maximum values (maximum can be >1) for each channel across all pixels 
        Step 2: Scales the values for each channel to the range of [0, 1]
    """
    min_vals = np.min(image, axis=(0, 1))
    max_vals = np.max(image, axis=(0, 1))
    normalised_image = (image - min_vals) / (max_vals - min_vals)
    return normalised_image


def clip_image(image):
    """ Function to clip the pixels in the image which are outside [0,1]
        If a pixel value is greater than 1, it is set to 1.
        If a pixel value is less than 0, it is set to 0.
    """
    clipped_image = np.clip(image, 0, 1)
    return clipped_image
