
from logging import raiseExceptions
from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import print_assert_equal


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
    
    f = cv2.imread(os.getcwd()+'\images'+'\cameraman.tif',flags=2)  # load the images
    f1 = cv2.imread(os.getcwd()+'\images'+'\einstein.tif',flags=2)  
    f2 = cv2.imread(os.getcwd()+'\images'+'\lena512color.tiff',flags=1) 
    f3 = cv2.imread(os.getcwd()+'\images'+'\mandril_color.tif',flags=1)
    f2 = rgb1gray(f2)
    f3 = rgb1gray(f3)  # transfer the colorful image into the gray ones 

    # task-1,test the convolution in different sigmas
    images = [f,f,f,f,f, f1,f1,f1,f1,f1, f2,f2,f2,f2,f2, f3,f3,f3,f3,f3]  # the images
    sigmas = [0,1,2,3,5, 0,1,2,3,5, 0,1,2,3,5, 0,1,2,3,5]
    titles = ['origin','sig=1','sig=2','sig=3','sig=5']  # the titles
    
    for i in range(20):
        plt.subplot(4,5,i+1)
        if sigmas[i] != 0:
            w = gaussKernel(sigmas[i])
            images[i] = twodConv(images[i],w)
        plt.imshow(images[i],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i>=15:
            plt.xlabel(titles[i-15])
    plt.show()
    
    # task-2 the comparison between twodConv and CV2.GaussianBlur
    images_1 = [f,f,f,f,f1,f1,f1,f1,f2,f2,f2,f2,f3,f3,f3,f3]
    titles_1 = ['origin','twodConv','GaussianBlur','dis-abs']
    for i in range(16):
        plt.subplot(4,4,i+1)
        if i%4 ==1:
            w = gaussKernel(1)
            images_1[i] = twodConv(images_1[i],w)
        elif i%4 == 2:
            images_1[i] = cv2.GaussianBlur(images_1[i],(7,7),1)
        elif i%4 == 3:
            images_1[i] = np.abs(images_1[i-1]-images_1[i-2])
        plt.imshow(images_1[i],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i>=12:
            plt.xlabel(titles_1[i-12])
    plt.show()
    
    # task-3 the comparison between two types of paddings
    images_2 = [f,f,f,f,f2,f2,f2,f2]
    titles_2 = ['origin','replicate padding','zero padding']
    for i in range(6):
        plt.subplot(2,3,i+1)
        
        if i%3 == 1:
            w = gaussKernel(1)
            images_2[i] = twodConv(images_2[i],w,'replicate')
        elif i%3 == 2:
            w = gaussKernel(1)
            images_2[i] = twodConv(images_2[i],w,'zero')
        if i >= 3:
            plt.xlabel(titles_2[i-3])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images_2[i],cmap='gray')
    plt.show()

    # task-3-1 the comparison between two types of paddings ('sig=1','sig=2','sig=3','sig=5')
    images_3 = [f,f,f,f,f, f,f,f,f,f, f2,f2,f2,f2,f2, f2,f2,f2,f2,f2]
    titles_3y = ['replicate','zero','replicate','zero']
    titles_3x = ['origin','sig=1','sig=2','sig=3','sig=5']
    sigmas_3 = [0,1,2,3,5, 0,1,2,3,5, 0,1,2,3,5, 0,1,2,3,5]
    methods_3 = ['replicate','replicate','replicate','replicate','replicate',\
        'zero','zero','zero','zero','zero',\
        'replicate','replicate','replicate','replicate','replicate',\
        'zero','zero','zero','zero','zero']
    for i in range(20):
        plt.subplot(4,5,i+1)
        if sigmas_3[i] != 0:
            w = gaussKernel(sigmas_3[i])
            images_3[i] = twodConv(images_3[i],w,methods_3[i])
        if not i%5:
            plt.ylabel(titles_3y[int(i/5)])
        if i >= 15:
            plt.xlabel(titles_3x[i-15])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images_3[i],cmap='gray')
    plt.show()
            

    
    
    

    






