import numpy as np
import cv2 as cv
import imutils
  
# path
path = "sample_handwriting/001.jpg"
path = 'templates/I.jpg'

#----Filtering stuffs-------
# Using cv2.imread() method
img = cv.imread(path,0)
median = cv.medianBlur(img,3)
median = cv.medianBlur(median,5)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
ret,thresh5 = cv.threshold(opening,0,1,cv.THRESH_TOZERO)

#----------template matching---------------------

template = cv.imread('templates/I.jpg', 0)# Read the template

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

threshold = 0.5 # Specify a threshold
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
	# resize the image according to the scale, and keep track
	# of the ratio of the resizing
    resized = imutils.resize(template, width = int(template.shape[1] * scale))
    r = template.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv.Canny(resized, 50, 200)
    result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
    # check to see if the iteration should be visualized
    # draw a bounding box around the detected region
    clone = np.dstack([edged, edged, edged])
    cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
        (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
    cv.imshow("Visualize", clone)
    cv.waitKey(0)
    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)



#----------Display Testing-----------------------
# erosion = cv.erode(img,kernel,iterations = 5)
# Displaying the image

cv.imshow('image', img)
cv.waitKey()

cv.imshow('image', thresh5)
cv.waitKey()

