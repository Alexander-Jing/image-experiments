from logging import raiseExceptions
from matplotlib import image, lines
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
    
    f = f.astype(np.float64)  # change the tpye of value in order to do the precise calculate 
    
    b,g,r=cv2.split(f)  # extract the b g r part from the image
    if method=='average':
        g = (np.asarray(b)+np.asarray(g)+np.asarray(r))/3  # the average method
        g = g.astype(np.uint8)
    elif method=='NTSC':
        g = (0.1140*np.asarray(b)+0.5870*np.asarray(g)+0.2989*np.asarray(r))
        g = g.astype(np.uint8)
    return g

if __name__ == "__main__":
    
    # test the function rgb1gray
    f = cv2.imread(os.getcwd()+'\images'+'\mandril_color.tif',flags=1)
    f1 = cv2.imread(os.getcwd()+'\images'+'\lena512color.tiff',flags=1)
    
    images = [f,f,f,f1,f1,f1]
    methods = ['colorful','average','NTSC','colorful','average','NTSC']
    titles = ['mandril-color','mandril-average','mandril-NTSC','lena-color','lena-average','lena-NTSC']
    
    for i in range(6):
        plt.subplot(2,3,i+1)
        if methods[i] == 'colorful':
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(rgb1gray(images[i],methods[i]),cmap='gray')
        plt.xlabel(titles[i])
    plt.show()
   
    