import numpy as np
import cv2 as cv
from linedraw import *
import random

# path
# path = "sample_handwriting/001.jpg"
path_1 = "images/gt_002.jpg"
path_2 = "images/002.jpg"

#----Filtering stuffs-------
img_1 = cv.imread(path_1,0)
median_1 = cv.medianBlur(img_1,3)

img_2 = cv.imread(path_2,0)
median_2 = cv.medianBlur(img_2,3)
# median = cv.medianBlur(median,5)
# kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
# ret,thresh5 = cv.threshold(opening,0,1,cv.THRESH_TOZERO)

def find_contours(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    return contours
    
def draw_contours(img,contours,window_name = "contours"):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR )
    for i in range(len(contours)):
        cv.drawContours(img, contours, i, 
                        (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                        ,3)
    cv.imshow(window_name, img)
#--------vectorize image----------
edges = cv.Canny(img_1,100,200)
invert_edges = cv2.bitwise_not(edges)

#--------display image------------
cv.imshow('median blur', median_1)
cv.waitKey()

cv.imshow('edges',invert_edges)
cv.waitKey()

contours_1 = find_contours(median_1)
contours_2 = find_contours(median_2)

draw_contours(img_1,contours_1,window_name = "contours 1")
draw_contours(img_2,contours_2,window_name = "contours 2")
# draw_contours(img_2,contours_2)

# cv.imshow('contours1', img_1)
# cv.imshow('contours2', img_2)
cv.waitKey()