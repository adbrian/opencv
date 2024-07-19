import cv2
import pytesseract
from PIL import Image
import numpy as np


image1 = cv2.imread('./1.jpg', 0)

def rescaleFrame(frame, scale):
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Adaptive Thresholding with Noise Reduction
def preprocess(frame):

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(frame)

    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove noise using median blur
    denoised = cv2.medianBlur(otsu, 3)

    return denoised

preimg = preprocess(image1)

cv2.imwrite('preprocessed_image1.jpg', preimg)

re_preprocessedimage = rescaleFrame(preimg, 0.20)

# cv2.imshow('Image', image1)
# cv2.imshow('Resized Image', re_image1)
# cv2.imshow('Resized Grayscale1', re_gray1)
# cv2.imshow('Resized Grayscale2', re_gray2)
# cv2.imshow('Resized Binary', re_binary1)
cv2.imshow('rescaled_Processed', re_preprocessedimage)
cv2.waitKey(5000)
