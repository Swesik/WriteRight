import numpy as np
import cv2 as cv
import random

def get_contour(letter,draw = False):
    path = "letter_templates/" + letter + ".jpg"
    print(path)
    img = cv.imread(path,0)
    
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    if draw:
        cv.imshow(letter,img)
        cv.waitKey()
        draw_contours(img,contours,letter)
    return contours

def draw_contours(img,contours,window_name = "contours"):
    h,w = img.shape
    vis = np.ones((h, w), np.float32)
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    # print(vis.shape)
    # print(scale_contour(contours[1],2))
    # cv.drawContours(vis, (contours[0],scale_contour(contours[1],2)), -1, 
    #                     (0,0,1)
    #                     ,6)
    cv.drawContours(vis,contours,2,(0,0,0),6)
    cv.imshow(window_name, vis)
    cv.waitKey()

def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


# ret = cv.matchShapes(cnt1,cnt2,1,0.0) ##matching contours?? 
# print( ret )

contours = get_contour("g",draw = True)

print(len(contours[0]))
