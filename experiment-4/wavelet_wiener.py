import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_noise(img, sigma):
    """ add noise for the image
    add noise for testing the wavelet-wiener

    args:
        img(np.array): the input image
        sigma(float): the sigma of the normal noise
    
    return:
        img(np.array): the output image with noise
    """
    row, col = img.shape
    img_noise = img + np.random.randn(row,col)*sigma  # add the gaussin noise
    img_noise = np.where(img_noise <= 0, 0, img_noise)
    img_noise = np.where(img_noise > 255, 255, img_noise)

    return np.uint8(img_noise)


def wiener_filter(img,HH):
    """ wiener_filter
    wiener_filter of the wavelet-wiener part

    args:
        img(np.array): the input image Y(i) with noise
        HH(np.array): the image HH with sigma of the normal noise in the wavelet composition 
    
    return:
        filter_image(np.array): X(i) the output image (in the process of wavelet transform)
    """
    noise_sigma = (np.median(np.abs(HH))/0.6745)**2  # according to the PPT, estimate the sigma_n
    img_sigma = 1/(img.shape[0]*img.shape[1])*(np.sum(img*img)-noise_sigma)  # calculate the sigma**2
    filter_image = img_sigma/(img_sigma+noise_sigma)*img  # calculate the output Xi after wiener filter 
    
    return filter_image



def wiener_wavelet(img,times=5):
    """wiener in the wavelet space
    the wiener filter in the wavelet space based on a pyramid structure 

    args:
        img(np.array): the input image
        times(int): times of wiener filtering 
    
    return:
        filter_image(np.array): the output image without noise
    """
    img = pywt.dwt2(img, 'bior4.4')
    LL, (LH, HL, HH) = img
    LH = wiener_filter(LH, HH)
    HL = wiener_filter(HL, HH)
    HH = wiener_filter(HH, HH)
    if times>0:
        LL = wiener_wavelet(LL,times-1)
        row, col = LL.shape
        # take care, the wavelet pywt may affect the image's shape, try to fix it
        # referring to https://blog.csdn.net/qq_36293056/article/details/113575634
        d_row = row - HH.shape[0]
        d_col = col - HH.shape[1]
        if d_row > 0 or d_col > 0:
            d_row = row - np.arange(d_row) - 1
            d_col = col - np.arange(d_col) - 1
            LL = np.delete(LL, d_row, axis=0)
            LL = np.delete(LL, d_col, axis=1)
    filter_image = pywt.idwt2((LL, (LH, HL, HH)), 'bior4.4')

    return filter_image



if __name__ == '__main__':

    img = cv2.imread('lena_gray_512.tif',cv2.IMREAD_GRAYSCALE)

    plt.subplot(1,3,1)    
    plt.imshow(img,cmap='gray')
    plt.xlabel("original image")  # the orignal image

    img_noise = add_noise(img, 20)
    plt.subplot(1,3,2)    
    plt.imshow(img_noise,cmap='gray')
    plt.xlabel("image with noise")  # the image with noise

    img_filter = wiener_wavelet(img_noise,4)
    img_filter = np.where(img_filter < 0, 0, img_filter)
    img_filter = np.where(img_filter > 255, 255, img_filter)
    img_filter = np.uint8(img_filter)
    plt.subplot(1,3,3)    
    plt.imshow(img_filter,cmap='gray')
    plt.xlabel("image after filtering")  # the image after filtering 
    
    plt.show()
