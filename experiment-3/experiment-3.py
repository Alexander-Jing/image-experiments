import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from numpy.core.fromnumeric import shape

def imbin(f,threshold=128):
    """image binarization function 
    using the threshold value method

    args:
        f (array): the input gray image
        threshold(int,default=128): the threshold 
    returns:
        the binary image
    """

    f = f.astype(np.float32)
    f = np.where(f>threshold,255,0)

    return f
    
def imEro(f,w = np.array([[0,1,0],[1,0,1],[0,1,0]])):
    """the erosion of the image (using the method of convolution)
    using the convolution methon to make the erosion of the iamge, using the zero-padding\
        take care in the image, 255 means the white background, while 0 means the image context, \
            so we have to transverse the image in order to make the Erosion  

    args:
        f(array): the original image
        w(array, \
            default w = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for erosion 
    
    return:
        f2(array): the image after erosion 
    """

    f = f.astype(np.float32)
    f = 255 - f  # in the image, 255 means the white background, while 0 means the image context, so we have to transverse the image in order to make the Erosion 
    
    pad_length,_ = w.shape
    p = int((pad_length-1)/2)  # calculate the size of padding  
    # use the zero padding 
    f2 = f.copy()
    f1 = np.vstack((np.zeros((p,f.shape[1])),f,np.zeros((p,f.shape[1]))))
    f1 = np.hstack((np.zeros((f1.shape[0],p)),f1,np.zeros((f1.shape[0],p))))  # the padding manipulation 
    
    for i in range(p,f.shape[0]-p):  # convolution 
        for j in range(p,f.shape[1]-p):
            if np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w) == np.sum(w)*255:
                f2[i,j] = 255
            else:
                f2[i,j] = 0
    return 255-f2

def imDil(f,w = np.array([[0,1,0],[1,0,1],[0,1,0]])):
    """the Dilation of the image (using the method of convolution)
    using the convolution methon to make the Dilation of the iamge, using the zero-padding\
        take care in the image, 255 means the white background, while 0 means the image context, \
            so we have to transverse the image in order to make the Dilation  

    args:
        f(array): the original image
        w(array, \
            w = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for Dilation 
    
    return:
        f2(array): the image after Dilation 
    """

    f = f.astype(np.float32)
    f = 255 - f  # in the image, 255 means the white background, while 0 means the image context, so we have to transverse the image in order to make the Erosion 
    
    pad_length,_ = w.shape
    p = int((pad_length-1)/2)  # calculate the size of padding  
    # use the zero padding 
    f2 = f.copy()
    f1 = np.vstack((np.zeros((p,f.shape[1])),f,np.zeros((p,f.shape[1]))))
    f1 = np.hstack((np.zeros((f1.shape[0],p)),f1,np.zeros((f1.shape[0],p))))  # the padding manipulation 
    
    for i in range(p,f.shape[0]-p):  # convolution 
        for j in range(p,f.shape[1]-p):
            if np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w) == 0:
                f2[i,j] = 0
            else:
                f2[i,j] = 255
    return 255-f2

def imOpen(f,w1 = np.array([[0,1,0],[1,0,1],[0,1,0]]),\
    w2 = np.array([[0,1,0],[1,0,1],[0,1,0]])):
    """opening operation digital image processing
    the first is the erosion operation and the second is the Dilation operation 

    args:
        f(array): the original image
        w1(array, default\
            w1 = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for erosion 
        w2(array, default\
            w2 = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for Dilation  
    
    returns:
        f2(array): the image after operation  
    """

    f = imEro(f,w1)
    f = imDil(f,w2)

    return f

def imSkeleton(f,w1 = np.array([[0,1,0],[1,0,1],[0,1,0]]),\
    w2 = np.array([[0,1,0],[1,0,1],[0,1,0]])):
    """the skeleton extraction through the morphology methods

    f(array): the original image
        w1(array, default\
            w1 = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for erosion 
        w2(array, default\
            w2 = np.array([[0,1,0],[1,0,1],[0,1,0]])): the kernel set for Dilation  

    returns:
        255 - SA_sum(array): the image after operation 
    """
    f = f.astype(np.float32)
    #f = 255 - f  # in the image, 255 means the white background, while 0 means the image context, so we have to transverse the image in order to make the Erosion 
    
    f1 = f.copy()
    f2 = f.copy()
    SA_sum = np.zeros(f.shape)

    while(True):

        f1 = imEro(f1,w1)
        if np.sum( (255 - f1)[1:f1.shape[0]-1,1:f1.shape[1]-1] ) == 0:
            break
        f2 = imOpen(f1,w1,w2)
        SA_sum += (255-f1) - (255-f2)
        SA_sum = np.where(SA_sum>255,255,SA_sum)
    
    return  255 - SA_sum

def imskeleton_dis(f,a,b):
    """the skeleton extraction method based on distance 
    
    args:
        f(array): the original image
        a,b(int): the parameter,1< b/a <2 
    return:
        the image after operation  
    """

    f = np.where(f==0,50,f)
    #f = np.where(f>250,128,f)
    f = 255 - f

    for i in range(2,f.shape[0]-1):
        for j in range(2,f.shape[1]-1):
            f[i,j] = min(f[i,j], f[i,j-1]+a, f[i-1,j-1]+b, f[i-1,j]+a, f[i-1,j+1]+b)
    
    for i1 in range(f.shape[0]-2,1,-1):
        for j1 in range(f.shape[1]-2,1,-1):
            f[i1,j1] = min(f[i1,j1], f[i1,j1+1]+a, f[i1+1,j1-1]+b, f[i1+1,j1]+a, f[i1+1,j1+1]+b)
    
    
    f = imbin(255 - f,200)  # because the output image is in the form of gray image, in order to make it much more correct, we use the binary image of the gray one
    
    return f

def imHMT(f,w = np.array([[0,0,0],[0,0,0],[0,0,0]])):
    """HMT 
    the Hit Miss Transform ,HMT, reference: the González's Digital Image Processing 

    args:
        f(array): the input image, using the zero padding
        w(array): the erosion kernel, if you would like to set the don't care 'x', set the x as 2
    return:
        the image after the HMT operation 
    """
    f = f.astype(np.float32)
    f = 255 - f  # in the image, 255 means the white background, while 0 means the image context, so we have to transverse the image in order to make the Erosion 
    
    pad_length,_ = w.shape
    p = int((pad_length-1)/2)  # calculate the size of padding  

    # use the zero padding 
    f2 = f.copy()
    f1 = np.vstack((np.zeros((p,f.shape[1])),f,np.zeros((p,f.shape[1]))))
    f1 = np.hstack((np.zeros((f1.shape[0],p)),f1,np.zeros((f1.shape[0],p))))  # the padding manipulation
    
    for i in range(p,f.shape[0]-p):  # convolution 
        for j in range(p,f.shape[1]-p):
            if np.sum(f1[i-p:i+p+1,j-p:j+p+1]== w*255) == (pad_length**2 - np.sum(w==2)):  # the MHT is similar to the Template match
                f2[i,j] = 255  
            else:
                f2[i,j] = 0
    
    return 255-f2

def imClipping(f,w_all =[np.array([[2,0,0],[1,1,0],[2,0,0]]),np.array([[2,1,2],[0,1,0],[0,0,0]]),\
        np.array([[0,0,2],[0,1,1],[0,0,2]]),np.array([[0,0,0],[0,1,0],[2,1,2]]),\
            np.array([[1,0,0],[0,1,0],[0,0,0]]),np.array([[0,0,1],[0,1,0],[0,0,0]]),\
                np.array([[0,0,0],[0,1,0],[0,0,1]]),np.array([[0,0,0],[0,1,0],[1,0,0]])]):
    """the image clipping 
    following the methods in the book Digital Image Processing

    args:
        f(array): original image
        w_all(array): the kernel used in the clipping 
    returns:
        the image after clipping  
    """
    x1 = f.copy()
    x2 = np.zeros(f.shape)
    x3 = np.zeros(f.shape)

    # refining the image
    for _ in range(3):
        for w in w_all:
            x1 = 255 - ((255-x1) - (255-imHMT(x1,w)))  # the care, the ouput is the white(255), black(0)
    # detecting the points
    for w in w_all:
        x2 += 255 - imHMT(x1,w)
    x2 = 255 - x2
    # the Dilation
    for _ in range(3):
        x3 = 255 - imDil(x2,w=np.array([[1,1,1],[1,1,1],[1,1,1]]))
        x3 = (x3/255 * (255-f)/255)*255
    x3 = 255 - x3
    # output
    x4 = (255-x1) + (255-x3)
    x4 = np.where(x4>255,255,x4)
    return 255-x4

    


    



if __name__=="__main__":
    
    
    plt.subplot(1,2,1)    
    f = cv2.imread("fingerprint.jpg",flags=2)
    #f = cv2.imread("lena_gray_512.tif",flags=2)
    f = np.asarray(f)
    
    f = imbin(f)
    plt.imshow(f.astype(np.uint8),cmap='gray')
    #plt.show()
    
    plt.subplot(1,2,2)
    #f = imEro(f,w)
    #f = imDil(f)
    #f = imOpen(f)
    #f = imSkeleton(f)
    #f = imskeleton_dis(f,a=45,b=60)
    w_all =[np.array([[2,0,0],[1,1,0],[2,0,0]]),np.array([[2,1,2],[0,1,0],[0,0,0]]),\
        np.array([[0,0,2],[0,1,1],[0,0,2]]),np.array([[0,0,0],[0,1,0],[2,1,2]]),\
            np.array([[1,0,0],[0,1,0],[0,0,0]]),np.array([[0,0,1],[0,1,0],[0,0,0]]),\
                np.array([[0,0,0],[0,1,0],[0,0,1]]),np.array([[0,0,0],[0,1,0],[1,0,0]])]
    
    f1 = imClipping(f,w_all)
    plt.imshow((f-f1).astype(np.uint8),cmap='gray')
    plt.show()