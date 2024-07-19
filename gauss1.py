import cv2
import pytesseract
from PIL import Image
import numpy as np


image1 = cv2.imread('./1.jpg')

# convert image to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# convert grayscale to binary
_, binary1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)




def rescaleFrame(frame, scale):
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# re_image1 = rescaleFrame(image1, 0.20)
re_gray1 = rescaleFrame(gray1, 0.20)
# re_binary1 = rescaleFrame(binary1, 0.20)

gray2 = cv2.imread('./1.jpg', 0)  # Read as grayscale
re_gray2 = rescaleFrame(gray2, 0.20)

thresh = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.ones((2,2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

dilation = cv2.dilate(opening,kernel, iterations=1)
re_dilated = rescaleFrame(dilation, 0.25)

cv2.imwrite('enhanced_receipt.jpg', dilation)
# cv2.imshow('Image', image1)
# cv2.imshow('Resized Image', re_image1)
# cv2.imshow('Resized Grayscale1', re_gray1)
# cv2.imshow('Resized Grayscale2', re_gray2)
# cv2.imshow('Resized Binary', re_binary1)
cv2.imshow('Processed', re_dilated)
cv2.waitKey(5000000)
