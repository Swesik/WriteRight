import cv2 as cv
import numpy as np
import random
import json
class Bubble:
    def __json_to_dict(self, file_name):
        with open(file_name) as json_file:
            bubbles_dict = json.load(json_file)
        to_ret = {}
        for key,val in bubbles_dict.items():
            for i in range(len(val)) :
                val[i] = np.array(val[i])
            val = tuple(val)
            to_ret[key] = val
        return to_ret
    
    def __init__(self, json_file):
        self.bubble_dict = self.__json_to_dict(json_file)
    
    def get_cnt(self,char):
        return self.bubble_dict[char]
    
    def get_norm_cnt(self,cnt,return_shift = False):
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
        
    def move_bubble(self,cnt,coord):
        cnt_moved = [0 for i in range(len(cnt))]
        for i in range(len(cnt)):
            cnt_moved[i] = (cnt[i] + coord).astype(np.int32)
        cnt_moved = tuple(cnt_moved)
        return cnt_moved
    
    def scale_cnt(self,cnt, scale):
        # cx,cy,cnt_norm = move_contour_to_origin(cnt, return_shift=True)
        cnt_scaled = [0 for i in range(len(cnt))]
        for i in range(len(cnt)):
            cnt_scaled[i] = cnt[i] * scale
        return cnt_scaled
        
    def __cart2pol(self, x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho
    
    def __pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
        
    def rotate_cnt(self, cnt, angle):
        # a_rotated = bubble.rotate_cnt(a_cnt, 90)
        # a_rotated = bubble.move_bubble(a_rotated,(500,500))
        cnt_rotated= [np.zeros_like(cnt[i]) for i in range(1,len(cnt))]
        for i in range(1, len(cnt)):
            coordinates = cnt[i][:, 0, :]
            xs, ys = coordinates[:, 0], coordinates[:, 1]
            thetas, rhos = self.__cart2pol(xs, ys)
            
            thetas = np.rad2deg(thetas)
            thetas = (thetas + angle) % 360
            thetas = np.deg2rad(thetas)
        
            xs, ys = self.__pol2cart(thetas, rhos)
            
            cnt_rotated[i-1][:, 0, 0] = xs
            cnt_rotated[i-1][:, 0, 1] = ys
        return cnt_rotated
    
    def draw_cnt(self,img,cnt,window_name = 'cnt'):
        h,w = img.shape
        vis = np.ones((h, w), np.float32)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        cv.drawContours(vis,cnt,-1,(0,0,0),6)
        cv.imshow(window_name, vis)
        cv.waitKey()
        return vis

    def draw_cnt_colored(self,img,cnt,window_name = 'cnt'):
        h,w = img.shape
        vis = np.ones((h, w), np.float32)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        for i in range(len(cnt)):
            cv.drawContours(vis,cnt,i,(0,0,0),6)
        cv.imshow(window_name, vis)
        cv.waitKey()
    
    def get_extreme(self,cnt):
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        return (extLeft[0],extTop[1])
    
    def scale_to_norm(self,cnt):
        norm = self.get_norm_cnt(cnt)
        print(norm[0])
    # def get_prob(self,cnt):

    # def get_order(self,cntrs):
    #     xtrm_point_list = {}
    #     for i in range(1, len(cntrs)):
    #         xtrm_point_list[self.get_extreme(cntrs[i])] = cntrs[i]
    #     point_list = xtrm_point_list.keys()
    #     x_sorted = sorted(point_list,key=lambda x: x[0])
    #     y_sorted = sorted(point_list,key=lambda y: y[1])
    #     order_to_match = []

    #     while (len(x_sorted) != 0):

    
def main():
    bubble = Bubble("bubbles.json")

    
    img = cv.imread("images/001.jpg",0)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    cnt, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt_norm = bubble.get_norm_cnt(cnt)
    cnt_scaled = bubble.scale_cnt(cnt_norm,2)
    to_draw_1 = bubble.move_bubble(cnt_scaled,(500,500))
    
    img = cv.imread("images/001.jpg",0)
    # vis = bubble.draw_cnt(img,to_draw_1)
    
    #new_img = cv.cvtColor(np.ones_like(img), cv.COLOR_GRAY2BGR)
    point_list = [bubble.get_extreme(i) for i in to_draw_1]

    #draw bubbles with dots
    # for i in range(1, len(to_draw_1)):
    #     print(bubble.get_extreme(to_draw_1[i]))
    #     cv.circle(vis, bubble.get_extreme(to_draw_1[i]), 8, (0, 0, 255), -1)
    # cv.imshow("window", vis)
    # cv.waitKey()

    bubble.scale_to_norm(to_draw_1)


main()