
from logging import raiseExceptions
from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import print_assert_equal

def scanLine4e(f,I,loc):
    """Fetches the pixels from the line or column of a picture 

    use the CV2 to load the images, and then extract the value of pixels 
    
    Args:
        f: numpy array, the picture with in the from of grey value 
        I: the integer(int), indicating the number of the row or column we want
        loc: the string, indicating the whether it is row or column, \
            " row" means in the row, "column" means in the column 
    
    Returns:
        s: numpy array, the grey value of the row or column in the picture
    
    Raises: 
        ValueError: the arg f must be the picture 
        ValueError: the loc must be 'row' or 'col'
    """
    f = np.asarray(f)  # transform the picture as a numpy array
    try:
        image_row,image_col = f.shape
        if image_row <= 1  or image_col <= 1:
            raise ValueError("f must be the image")  # raise the error if it is not an image 
    except:
        raise
    
    try:
        if loc!='row' and loc!='col':
            raise ValueError("loc must be 'row' or 'col'")  # raise the error if the arg loc is not 'row' or 'col'
    except:
        raise
    
    if loc=='row':
        s = f[I,:]
    elif loc=='col':
        s = f[:,I]
    
    return s    

def rgb1gray(f,method='NTSC'):
    """transform the colorful image to the grey image

    two ways of transforming, the first is that 'average' way, the second is the 'NTSC' way, \
        two methods can be taken in the function 

    Args: 
        f: numpy array, the colorful picture
        method: the string, if it is 'average', use the average way, if it is 'NTSC', use the NTSC way\
            'NTSC' willed be set as default 
    
    Returns:
        g: the grey picture 
    
    Raises:
        ValueError: the arg f must be the picture 
        ValueError: method must be 'average' or 'NTSC' 

    """
    f = np.asarray(f)  # transform the picture as a numpy array
    try:
        if len(f.shape) < 3 :
            raise ValueError("f must be the colorful image")  # raise the error if it is not an image 
    except:
        raise
    
    try:
        if method!='average' and method!='NTSC':
            raise ValueError("method must be 'average' or 'NTSC'")  # raise the error if the arg method is not 'average' or 'NTSC'
    except:
        raise
    
    b,g,r=cv2.split(f)  # extract the b g r part from the image
    if method=='average':
        g = (np.asarray(b)+np.asarray(g)+np.asarray(r))/3  # the average method
        g = g.astype(np.uint8)
    elif method=='NTSC':
        g = (0.1140*np.asarray(b)+0.5870*np.asarray(g)+0.2989*np.asarray(r))
        g = g.astype(np.uint8)
    return g

def twodConv(f,w,method='zero'):
    """convolution kernel and padding skills

    the convolution kernel and two types of padding skills, the "replicate" skill and the "zero" skill 

    Args:
        f: numpy array, the grey picture 
        w: numpy array, the convolution kernel
        method: string, if it is "replicate", use the replicate padding, otherwise if it is "zero", use the zero padding\
            zero padding will be set as default 
    
    Returns:
        the convoluted image
    
    Raises:
        ValueError: the arg f must be the picture 
        ValueError: method must be 'replicate' or 'zero' 

    """
    f = np.asarray(f)  # transform the picture as a numpy array
    try:
        image_row,image_col = f.shape
        if image_row <= 1  or image_col <= 1:
            raise ValueError("f must be the image")  # raise the error if it is not an image 
    except:
        raise
    
    try:
        if method!='replicate' and method!='zero':
            raise ValueError("method must be 'replicate' or 'zero'")  # raise the error if the arg method is not 'replicate' or 'zero'
    except:
        raise
    pad_length,_ = w.shape
    p = int((pad_length-1)/2)  # calculate the size of padding 
    w = w.T  # transverse the convolution kernel matrix in order to simplify the calculation
    f2 = f.copy()
    if method == 'zero':
        f1 = np.vstack((np.zeros((p,f.shape[1])),f,np.zeros((p,f.shape[1]))))
        f1 = np.hstack((np.zeros((f1.shape[0],p)),f1,np.zeros((f1.shape[0],p))))  # the padding manipulation 
        for i in range(p,f.shape[0]-p):  # convolution 
            for j in range(p,f.shape[1]-p):
                f2[i,j] = np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w)
    if method == 'replicate':
        f1 = np.vstack((np.tile(f[0,:],(p,1)),f,np.tile(f[f.shape[0]-1,:],(p,1))))
        f1 = np.hstack((np.tile(np.reshape(f1[:,0],(-1,1)),(1,p)),f1,\
            np.tile(np.reshape(f1[:,f1.shape[1]-1],(-1,1)),(1,p))))  # the padding manipulation 
        for i in range(p,f.shape[0]-p):  # convolution 
            for j in range(p,f.shape[1]-p):
                f2[i,j] = np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w)
    f2 = f2.astype(np.uint8)
    return f2

def gaussKernel(sig,m=None):
    """form the gaussin kernel 

    general the gaussin kernel, the sig is the sigma in definition, the kernel will be normalized finally 

    Args:
        sig: float, the parameter sigma in the definition of guassin distribution 
        m: integer, the size of the kernel 

    returns:
        w: numpy array, the kernel 

    raises:
        ValueError: the arg sigma must be larger than zero 
        ValueError: m must be not less than one

    """
    try:
        if sig <= 0 :
            raise ValueError("the arg sigma must be larger than zero")
    except:
        raise
    
    if m == None:
        m = math.ceil(3*sig)*2 + 1  # set the kernel size

    try:
        if int(m) < 1:
            raise ValueError("m must be not less than one")
    except:
        raise
    
    w = np.eye(m)  # initial the kernel w
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] = 1/(2*np.pi*sig*sig)*\
                np.exp(-( (i-(w.shape[0]-1)/2)**2/(2*sig**2) + (j-(w.shape[1]-1)/2)**2/(2*sig**2)))  # take care, the kernel of gaussin fuction should change the center
    w = 1/np.sum(w) * w  # normalization
    
    return w

    


if __name__ == "__main__":
    """
    # test the function scanLine4e(f,I,loc)
    f = cv2.imread(os.getcwd()+'\images'+'\cameraman.tif',flags=2)
    f = np.asarray(f)
    image_val = scanLine4e(f,125,'row')
    x_aixs = np.array([i for i in range(image_val.shape[0])])
    plt.plot(x_aixs,image_val)
    plt.show()"""

    """# test the function rgb1gray
    f = cv2.imread(os.getcwd()+'\images'+'\mandril_color.tif',flags=1)
    cv2.imshow('colorful',f)  # the initial colorful image 
    f = rgb1gray(f)
    cv2.imshow('grey',f)
    cv2.waitKey(0)
    # test the function twodConv
    w = np.eye(3)
    f = twodConv(f,w)
    cv2.imshow('conv',f)
    cv2.waitKey(0)
    # test the kernel
    w = gaussKernel(1)
    print(w.shape)"""

    # test the gaussin filter
    #f = cv2.imread(os.getcwd()+'\images'+'\mandril_color.tif',flags=1)  #'\lena512color.tiff' \mandril_color.tif
    #cv2.imshow('colorful',f)  # the initial colorful image 
    #print(f.shape)
    #f = rgb1gray(f)  # the grey image 
    
    f = cv2.imread(os.getcwd()+'\images'+'\einstein.tif',flags=2)  #\einstein.tif \cameraman.tif
    cv2.imshow('grey',f)
    f_gau = cv2.GaussianBlur(f,(7,7),1)
    cv2.imshow('gaussinblur',f_gau)
    #print(f.shape)
    w = gaussKernel(1)  # the filter and convolution 
    f = twodConv(f,w,'replicate')
    f_dis = np.abs(f-f_gau)
    #print(f.shape)
    cv2.imshow('conv',f)
    cv2.imshow('dis',f_dis)
    cv2.waitKey(0)

    
    
    

    






