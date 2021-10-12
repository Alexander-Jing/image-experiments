
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def scanLine4e(f,I,loc):
    """Fetches the pixels from the line or column of a picture 

    use the CV2 to load the images, and then extract the value of pixels 
    
    Args:
        f: the picture with in the from of grey value 
        I: the integer(int), indicating the number of the row or column we want
        loc: the string, indicating the whether it is row or column, " row" means in the row, "column" means in the column 
    
    Returns:
        s: the list, the grey value of the row or column in the picture
    
    Raises: 
        ValueError: the arg f must be the picture 
    """
    try:
        image_row,image_col = f.shape
        if image_row <= 1  or image_col <= 1:
            raise ValueError("f must be the image")  # raise the error if it is not an image 
    except:
        raise

f = cv2.imread(os.getcwd()+'\images'+'\lena512color.tiff',flags=1)
f = np.asarray(f)
f = np.asarray(f)
print(len(f.shape))
"""print(f[0,0,:])
grey = (f[:,:,0]+f[:,:,1]+f[:,:,2])/3
grey = grey.astype(np.uint8)
cv2.imshow('grey',grey)
cv2.waitKey(0)

b,g,r=cv2.split(f)
grey = (np.asarray(b)+np.asarray(g)+np.asarray(r))/3
grey = grey.astype(np.uint8)
cv2.imshow('grey1',grey)
cv2.waitKey(0)
"""
b,g,r=cv2.split(f)
f = (0.1140*np.asarray(b)+0.5870*np.asarray(g)+0.2989*np.asarray(r))
f = f.astype(np.uint8)
cv2.imshow('grey2',f)
cv2.waitKey(0)
print(f.shape)



f2 = f.copy()
w = np.ones([5,5])
pad_length,_ = w.shape
p = int((pad_length-1)/2)  # calculate the size of padding
"""f1 = np.vstack((np.zeros((p,f.shape[1])),f,np.zeros((p,f.shape[1]))))
f1 = np.hstack((np.zeros((f1.shape[0],p)),f1,np.zeros((f1.shape[0],p))))
print(f1.shape)
for i in range(1,f.shape[0]-1):
    for j in range(1,f.shape[1]-1):
        f2[i,j] = np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w)
print(f2.shape)
f2 = f2.astype(np.uint8)
cv2.imshow('grey3',f2)
cv2.waitKey(0)"""

f1 = np.vstack((np.tile(f[0,:],(p,1)),f,np.tile(f[f.shape[0]-1,:],(p,1))))
f1 = np.hstack((np.tile(np.reshape(f1[:,0],(-1,1)),(1,p)),f1,np.tile(np.reshape(f1[:,f1.shape[1]-1],(-1,1)),(1,p))))
f1 = f1.astype(np.uint8)
cv2.imshow('grey3',f1)
cv2.waitKey(0)
print(f1.shape)
for i in range(p,f.shape[0]-p):
    for j in range(p,f.shape[1]-p):
        f2[i,j] = np.sum(f1[i-p:i+p+1,j-p:j+p+1]*w)

f2 = f2.astype(np.uint8)
cv2.imshow('grey4',f2)
cv2.waitKey(0)
