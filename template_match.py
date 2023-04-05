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
    
    def __get_height(self, cnt):
        transpose= np.transpose(cnt)
        height = np.amax(transpose, 2) - np.amin(transpose, 2)
        return height[1]
    
    def __max_height(self, cntrs):
        # Calculate maximum rectangle height
        max_height = 0
        for i in range(1, len(cntrs)):
            h = self.__get_height(cntrs[i])
            if h > max_height: max_height = h
        print(max_height)
        return max_height

    def sort_cntrs(self, cntrs):
        max_height = self.__max_height(cntrs)
        # Sort the contours by y-value
        by_y = self.__y_sorted(cntrs)
        print(by_y)
        line_y = list(by_y.keys())[0][1]      # first y
        line = 1
        by_line = {1:[]}

        # Assign a line number to each contour
        for key in by_y:
            y = key[1]
            if y > line_y + max_height:
                line_y = y
                line += 1
                by_line[line] = []
            by_line[line].append(key)
        # print(by_line)
        contours_sorted = [cntrs[0]]
        # This will now sort automatically by line then by x
        for key, val in by_line.items(): 
            x_sorted= sorted(val,key=lambda x: x[0])
            for point in x_sorted:
                contours_sorted.append(cntrs[by_y[point]])
        return contours_sorted

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
        transpose = np.transpose(norm[0])
        scaling_factor = np.min(np.amax(transpose,2) - np.amin(transpose,2))
        print(scaling_factor)
        return self.scale_cnt(cnt,2000/scaling_factor)
    
    def get_prob(self,cnt, metric = "shape"):
        to_ret = {}
        cnt_norm = self.scale_to_norm(cnt)
        for key,val in self.bubble_dict.items():
            if metric == "shape":
                val_norm = self.scale_to_norm(val)
                print(type(cnt))
                print(type(val))
                print(type(cnt_norm))
                print(type(val_norm))
                # to_ret[key] = cv.matchShapes(cnt[0],val[0],1,0.0)
                to_ret[key] = cv.matchShapes(cnt_norm[0],val_norm[0],1,0.0)
            elif metric == "area":
                to_ret[key] = cv.contourArea(val) - cv.contourArea(cnt_norm)
            else:
                return -1
        return to_ret
    
    def __y_sorted(self,cntrs):
        xtrm_point_list = {}
        for i in range(1, len(cntrs)):
            #xtrm_point_list[self.get_extreme(cntrs[i])] = cntrs[i]
            xtrm_point_list[self.get_extreme(cntrs[i])] = i
        point_list = xtrm_point_list.keys()
        #x_sorted = sorted(point_list,key=lambda x: x[0])
        y_sorted = sorted(point_list,key=lambda y: y[1])
        sorted_dict = {i: xtrm_point_list[i] for i in y_sorted}
        return sorted_dict
    
def main():
    bubble = Bubble("bubbles.json")
    
    img = cv.imread("images/001.jpg",0)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    cnt, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)\
    
    cnt_norm = bubble.get_norm_cnt(cnt)
    cnt_scaled = bubble.scale_cnt(cnt_norm,2)
    cnt_moved = bubble.move_bubble(cnt_scaled, (500, 500))
    #print(cnt_scaled)
    to_draw_1 = bubble.move_bubble(cnt_norm,(500,500))
    sorted_cnt = bubble.sort_cntrs(cnt_moved)
    #img = cv.imread("images/001.jpg",0)
    for cnt in sorted_cnt:
        vis = bubble.draw_cnt(img,cnt)
    
    #cv.circle(vis, (1500, 1500),8, (0, 255, 0) -1)
    # cv.circle(vis, (1500, 10), 8, (0, 0, 255), -1)
    # cv.imshow("hi", vis)
    #new_img = cv.cvtColor(np.ones_like(img), cv.COLOR_GRAY2BGR)
    #point_list = [bubble.get_extreme(i) for i in to_draw_1]

    #draw bubbles with dots
    # for i in range(1, len(to_draw_1)):
    #     print(bubble.get_extreme(to_draw_1[i]))
    #     cv.circle(vis, bubble.get_extreme(to_draw_1[i]), 8, (0, 0, 255), -1)
    # cv.imshow("window", vis)
    # cv.waitKey()
    bubble.get_prob(cnt)


main()