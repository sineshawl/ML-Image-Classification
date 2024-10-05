import numpy as np
import pywt
import cv2    

def toWavelet(img, mode='haar', level=1):
    imgArray = img
    #Datatype conversions
    #convert to grayscale
    imgArray = cv2.cvtColor( imgArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imgArray =  np.float32(imgArray)   
    imgArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imgArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imgArray_H=pywt.waverec2(coeffs_H, mode);
    imgArray_H *= 255;
    imgArray_H =  np.uint8(imgArray_H)
    return imgArray_H
