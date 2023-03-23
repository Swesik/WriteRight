import numpy as np
import cv2 as cv
# import imutils
  
# path
print("hi")
path = "sample_handwriting/001.jpg"
# path = 'templates/I.jpg'

#----Filtering stuffs-------
# Using cv2.imread() method
img = cv.imread(path,0)
median = cv.medianBlur(img,3)
median = cv.GaussianBlur(median,(5,5),0)
# median = cv.medianBlur(median,5)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
ret,thresh5 = cv.threshold(opening,0,1,cv.THRESH_TOZERO)

#----------Feature detection---------------------

# from matplotlib import pyplot as plt

# img1 = cv.imread('sample_handwriting/001.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
img1 = cv.imread('sample_handwriting/001.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('templates/I.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# plt.imshow(img3,),plt.show()

# plt.imshow(img3),plt.show()


#----------Display Testing-----------------------
# erosion = cv.erode(img,kernel,iterations = 5)
# Displaying the image

# cv.imshow('image', img)
# cv.waitKey()

cv.imshow("draw me dots", img3)
cv.waitKey()

cv.imshow('image', thresh5)
cv.waitKey()

