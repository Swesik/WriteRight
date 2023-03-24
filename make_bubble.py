import numpy as np
import cv2 as cv
import random
import json
import os
import re
from json import JSONEncoder

def get_contour(letter,draw = False):
    path = "letter_templates/" + letter + ".jpg"
    # print(path)
    img = cv.imread(path,0)
    
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print('num contours = ', len(contours))
    if draw:
        cv.imshow(letter,img)
        cv.waitKey()
        draw_contours(img,contours,letter)
    return contours

def draw_contours(img,contours,window_name = "contours",index = -1):
    h,w = img.shape
    vis = np.ones((h, w), np.float32)
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
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
    #example input: make_json("letter_templates", "bubbles.json")
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
    with open("bubbles.json", "w") as outfile:
        json.dump(dictionary, outfile)

def json_to_dict(file_name):
    with open(file_name) as json_file:
        bubbles_dict = json.load(json_file)
    to_ret = {}
    for key,val in bubbles_dict.items():
        for i in range(len(val)) :
            val[i] = np.array(val[i])
        val = tuple(val)
        to_ret[key] = val
        # print(contour_a[0].shape)
        # print(contour_a[1].shape)
        # print(contour_a[2].shape)
        # print(type(contour_a[0]))
    return to_ret

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

def main():
    make_json("letter_templates", "bubbles.json")
    img = cv.imread("letter_templates/k.jpg",0)
    bubble_dict = json_to_dict("bubbles.json")
    draw_contours(img, bubble_dict["\""], window_name="contour_\"")
    scaled_cnt = scale_contour(bubble_dict["\""],2)
    draw_contours(img, scaled_cnt, window_name="scl_contour_\"")
    moved_cnt = move_contour_to_anywhere(scaled_cnt, 500, 500)
    draw_contours(img, moved_cnt, window_name="mv_contour_\"")
    # scaled_cnt = scale_contour(bubble_dict["!"][2],2)
    # draw_contours(img, scaled_cnt, window_name="scl_contour_!_2")
main()