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
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                               iterations=2)
    
    return opening

preimg = preprocess(image1)

cv2.imwrite('preprocessed_image1.jpg', preimg)

re_preprocessedimage = rescaleFrame(preimg, 0.20)

# cv2.imshow('Image', image1)
# cv2.imshow('Resized Image', re_image1)
# cv2.imshow('Resized Grayscale1', re_gray1)
# cv2.imshow('Resized Grayscale2', re_gray2)
# cv2.imshow('Resized Binary', re_binary1)
cv2.imshow('rescaled_Processed', re_preprocessedimage)
cv2.waitKey(5000000)
