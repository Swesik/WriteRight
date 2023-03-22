import numpy as np
import cv2 as cv
from linedraw import *
import random

# path
# path = "sample_handwriting/001.jpg"
path = "images/sample001_exact_match.jpg"

#----Filtering stuffs-------
img = cv.imread(path,0)
median = cv.medianBlur(img,3)
# median = cv.medianBlur(median,5)
# kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
# ret,thresh5 = cv.threshold(opening,0,1,cv.THRESH_TOZERO)

ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#--------vectorize image----------
edges = cv.Canny(median,100,200)
invert_edges = cv2.bitwise_not(edges)

print(len(contours))

#--------display image------------
cv.imshow('median blur', median)
cv.waitKey()

cv.imshow('edges',invert_edges)
cv.waitKey()

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR )
for i in range(len(contours)):
    cv.drawContours(img, contours, i, 
                    (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                    ,3)

cv.imshow('contours', img)
cv.waitKey()