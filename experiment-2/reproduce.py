from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


def dft2D(f):
    """2DFFT cascading two 1D FFT  
    the fast Fourier transition in 2D, according to the homework requests,\
        the 2D FFT will be implemented by combining two 1D Fourier transform 

    Args:
        f: nparray float, the input gray image 

    returns:
        F: nparray complex, the FFT image in the form of complex 

    raises:
        ValueError: the arg f must be the picture 
    """
    
    f = f.astype(np.float64)  # transform the image into np.array form with np.float64
    f = np.asarray(f, dtype=complex)  # set the type of the image as the complex type, 
                                      #\in order to fit the Fourier transform  

    M,N = f.shape  # get the shape of the image 
    try:
        if M <= 1 or N <=1 :
            raise ValueError("f must be the image")  # raise the error if it is not an image 
    except:
        raise
    
    F = f.copy()  # prepare the FFT output F
    
    for u in range(M):
        F[u,:] = np.fft.fft(f[u,:],N,axis=0)
    for v in range(N):
        F[:,v] = np.fft.fft(F[:,v],M,axis=0)  # it can be demonstrated that the 2D fourier transform\
                                              # can be cascaded by two 1D fourier transform 

    return F        



def idft2D(F):
    """2D IFFT cascading two 1D IFFT 
    the inverse Fourier transform, using two 1D inverse Fourier transform

    Args:
        F: nparray complex, the image after the Fourier transform 
    
    returns:
        f: nparray float, the IFFT image, also the original image 

    raise:
        ValueError: the arg F must be the picture
        TypeError: the arg F must be the type of complex 
    """

    try:
        if F.dtype != 'complex128':
            raise TypeError("F must be in the type of complex")
    except:
        raise

    M,N = F.shape  # get the shape of the image 
    try:
        if M <= 1 or N <=1 :
            raise ValueError("F must be the image")  # raise the error if it is not an image 
    except:
        raise

    f = F.copy()
    
    # we use the FFT function to perform the IFFT manipulation, there is relationship between the FFT and IFFT 
    for x in range(M):
        f[x,:] = np.fft.fft(np.conj(F[x,:]),N,axis=0)  # according to the request of the homework and theories on the book, we use the conj of the input F 
    for y in range(N):
        f[:,y] = np.fft.fft(f[:,y],M,axis=0)  
    f = (1/(M*N))*np.conj(f)  # according to the formula, we have to do the np.conj and multiple the (1/MN)

    f = np.real(f)  
    f = np.abs(f)  # we choose the real part of the IFFT image 
    f.astype(np.float)  # take care, the image type is np.float, in order to make the farther manipulations
    
    return f




if __name__ == "__main__":
    
    plt.subplot(2,2,1)
    test_image = np.zeros([512,512])
    #test_image[227:287,248:266] = np.ones([60,18])
    #test_image[197:317,247:267] = np.ones([120,20])
    test_image[207:307,248:266] = np.ones([100,18])  # set the size of the white block 
    plt.imshow(test_image.astype(np.uint8),cmap='gray')
    
    plt.subplot(2,2,2)
    test_image_fft = dft2D(test_image)
    plt.imshow((np.abs(test_image_fft)),cmap='gray')
    
    plt.subplot(2,2,3)
    test_image_center = test_image.copy()
    for x in range(test_image_center.shape[0]):  # generate the center image 
        for y in range(test_image_center.shape[1]):
            test_image_center[x,y] = test_image_center[x,y]*(-1)**(x+y)  
    F = dft2D(test_image_center)
    plt.imshow(np.abs(F), cmap='gray')
    plt.subplot(2,2,4)

    S = np.log2(1+abs(F))
    plt.imshow(np.abs(S), cmap='gray')
    plt.show()
    


