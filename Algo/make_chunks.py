import numpy as np
import cv2 as cv
import random
import json
import os
import re
from json import JSONEncoder
import math

chunk_type = tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]

def get_contour(letter:str,draw = False) -> np.ndarray:
    # path = "skeeenny_letters/" + letter + ".jpg"
    path = "skinny_queen_letters/new_s.jpg"
    # path = "images/002.jpg"
    # print(path)
    img = cv.imread(path,0)
    # print(img.shape)
    # ret, thresh = cv.threshold(img, 127, 255, 0)
    canny = cv.Canny(img, 100,200)
    
    # sobel = cv.Sobel(img, -1,0,2,3)
    # laplace = cv.Laplacian(img,-1,3)
    cv.imshow("canny",canny)
    cv.waitKey()
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print('num contours = ', len(contours))
    if draw:
        cv.imshow(letter,img)
        cv.waitKey()
        draw_contours(img,contours,letter, scale = 5)
    return contours

def draw_contours(img,contours,window_name = "contours",index = -1,scale = 1):
    h,w = img.shape
    vis = np.ones((h, w), np.float32)
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    # contours = scale_contour(contours, scale)
    # print(vis.shape)
    # print(scale_contour(contours[1],2))
    # cv.drawContours(vis, (contours[0],scale_contour(contours[1],2)), -1, 
    #                     (0,0,1)
    #                     ,6)
    cv.drawContours(vis,contours,index,(0,0,0),6)
    cv.imshow(window_name, vis)
    cv.waitKey()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def make_json(folder_name, out_file_name):
    #example input: make_json("skinny_queen_letters", "chunks.json")
    dictionary = {}
    for filename in os.listdir(folder_name):
        character = re.search("^.*\.", filename).group(0)[:-1]
        f = os.path.join(folder_name, filename)
        # checking if it is a file
        if os.path.isfile(f):
            contour = get_contour(character)
            origin_contour = move_contour_to_origin(contour)
            dictionary[character] = [a.tolist() for a in origin_contour]
    # print(len(dictionary))
    with open("chunks.json", "w") as outfile:
        json.dump(dictionary, outfile)

def move_contour_to_origin(cnt,return_shift = False):
    cx = 0
    cy = 0
    tot_mass = sum([len(i) for i in cnt])
    for i in cnt:
        M = cv.moments(i)
        cx += len(i) * int(M['m10']/M['m00'])
        cy += len(i) * int(M['m01']/M['m00'])
    cx = int(cx/tot_mass)
    cy = int(cy/tot_mass)
    cnt_norm = [0 for i in range(len(cnt))]
    for i in range(len(cnt)):
        cnt_norm[i] = (cnt[i] - [cx, cy]).astype(np.int32)
    cnt_norm = tuple(cnt_norm)
    if return_shift:
        return cx,cy, cnt_norm
    else:
        return cnt_norm
    
def move_contour_to_anywhere(cnt, cx, cy):
    cnt_norm = move_contour_to_origin(cnt, False)
    cnt_moved = [0 for i in range(len(cnt))]
    for i in range(len(cnt_norm)):
        cnt_moved[i] = (cnt_norm[i] + [cx,cy]).astype(np.int32)
    cnt_moved = tuple(cnt_moved)
    return cnt_moved

def scale_contour(cnt, scale):
    cx,cy,cnt_norm = move_contour_to_origin(cnt, return_shift=True)
    cnt_scaled = [0 for i in range(len(cnt_norm))]
    for i in range(len(cnt_norm)):
        cnt_scaled[i] = cnt_norm[i] * scale
        cnt_scaled[i] = (cnt_scaled[i] + [cx, cy]).astype(np.int32)
    return cnt_scaled

def make_chunks(cntr: np.ndarray,box_height:float):
    # chunk = {ptA, ptB, ptC, ptD, slope}
    ro_ret = []
    for i in range(1,cntr):
        pt_1 = cntr[i-1]
        pt_2 = cntr[i]
        slope = (pt_2[1]-pt_1[1])/(pt_2[0]-pt_1[0])
        perp_slope = -1/slope
        mag = math.sqrt(perp_slope**2 + 1)
        unit_vec = np.array([[1/mag],[perp_slope/mag]])
        new_chunk = (pt_1+box_height * unit_vec,pt_1-box_height * unit_vec,pt_2+box_height * unit_vec,pt_2-box_height * unit_vec, slope)
        ro_ret.append(new_chunk)
    return ro_ret

def main():
    #TODO 1. Read the contours
    #     2. Make chunks 
    #     3. Find way to combine chunks(with slope tresholds)
    #     4. Determine which chunk a segment is in (using centroid)
    #     5. make a chunks class
    
    cnt = get_contour("s",True)
    for i in cnt:
        print(len(i))
    # print(len(cnt))
    chunks = make_chunks(cnt,5)
    # scaled_cnt = scaled_cnt(cnt,5)
    
    # make_json("skinny_queen_letters", "chunks.json")
    # img = cv.imread("letter_templates/k.jpg",0)
    #chunk_type = json_to_dict("chunk.json")
    # draw_contours(img, bubble_dict["\""], window_name="contour_\"")
    # scaled_cnt = scale_contour(bubble_dict["\""],2)
    # draw_contours(img, scaled_cnt, window_name="scl_contour_\"")
    # moved_cnt = move_contour_to_anywhere(scaled_cnt, 500, 500)
    # draw_contours(img, moved_cnt, window_name="mv_contour_\"")
    # scaled_cnt = scale_contour(bubble_dict["!"][2],2)
    # draw_contours(img, scaled_cnt, window_name="scl_contour_!_2")
main()