import numpy as np
import cv2 as cv

  
# path
path = "sample_handwriting/001.jpg"
  
# Using cv2.imread() method
img = cv.imread(path,0)
median = cv.medianBlur(img,3)
median = cv.medianBlur(median,5)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
ret,thresh5 = cv.threshold(opening,0,1,cv.THRESH_TOZERO)

# erosion = cv.erode(img,kernel,iterations = 5)
# Displaying the image
cv.imshow('image', img)
cv.waitKey()

cv.imshow('image', thresh5)
cv.waitKey()
