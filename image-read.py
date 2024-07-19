import cv2
import numpy as np
import pytesseract
import os

img1 = cv2.imread("1.jpg")
h,w,cc = img1.shape
img1 = cv2.resize(img1, (w//5, h//5))

orb = cv2.ORB_create(3000)
kp1, des1 = orb.detectAndCompute(img1, None)
impKp1 = cv2.drawKeypoints(img1, kp1, None)

cv2.imshow("Key", impKp1)
cv2.imshow("Image1", img1)
cv2.waitKey(50000)