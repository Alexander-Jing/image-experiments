from logging import raiseExceptions
from matplotlib import image, lines
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

if __name__ == "__main__":
    
    # test the function scanLine4e(f,I,loc)
    f = cv2.imread(os.getcwd()+'\images'+'\cameraman.tif',flags=2)
    f = np.asarray(f)
    f1 = cv2.imread(os.getcwd()+'\images'+'\einstein.tif',flags=2)
    f1 = np.asarray(f1)

    images = [f,f,f1,f1]  # record the images for plotting
    pos = [int(f.shape[0]/2),int(f.shape[1]/2),int(f1.shape[0]/2),int(f1.shape[1]/2)]  # record the lines for plotting
    lines_iamge = ['row','col','row','col']  # the args locs
    titles_image = ['cameraman row','cameraman col','einstein row','einstein col']  # the y labels

    for i in range(4):
        # plot the four pictures as mattered in the orders
        plt.subplot(2,2,i+1)
        image_val = scanLine4e(images[i], pos[i], lines_iamge[i])
        x_aixs = np.array([i for i in range(image_val.shape[0])])
        plt.xlabel(titles_image[i])
        plt.ylabel('gray value')
        plt.plot(x_aixs,image_val)
    
    plt.show()
    

