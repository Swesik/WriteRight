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
        for i in range(1, len(cntrs)-2):#for weird paper lines, get rid of last 2
            h = self.__get_height(cntrs[i])
            # print(h)
            if h > max_height: max_height = h
        print(max_height)
        return max_height

    def sort_cntrs(self, cntrs):
        max_height = self.__max_height(cntrs)
        # Sort the contours by y-value
        by_y = self.__y_sorted(cntrs)
        # print(by_y)
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
    
    def draw_cnt(self,img,cnt,index = -1, window_name = 'cnt'):
        h,w = img.shape
        vis = np.ones((h, w), np.float32)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        cv.drawContours(vis,cnt,index,(0,0,0),6)
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
    
    def __get_extreme(self,cnt):
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        return (extLeft[0],extTop[1])
    
    def __get_bubble_size(self, cntrs):
        #intakes a "letter" from bubble.dict (the whole list of contours for that letter)
        #loop through each cnt, find the extremeTop, save the most extreme top
        #^^, find extremeBottom, save the most extreme Bottom 
        minY = float('inf')
        minX = float('inf')
        maxY = float('-inf')
        maxX = float('-inf')
        for cnt in cntrs[1:]: #loop through everything except for bounding box
            extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            extBottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
            if (extTop[1] < minY): minY = extTop[1]
            if (extBottom[1] > maxY): maxY = extBottom[1]
            if (extLeft[0] < minX): minX = extLeft[0]
            if (extRight[0] < maxX): maxX = extRight[0]
        return min(maxX- minX, maxY- minY)
    
    def __scale_to_norm(self,cntrs):
        norm = self.get_norm_cnt(cntrs)
        scaling_factor = self.__get_bubble_size(cntrs)
        # transpose = np.transpose(norm[0])
        # scaling_factor = np.min(np.amax(transpose,2) - np.amin(transpose,2))
        # print(scaling_factor)
        return tuple(self.scale_cnt(cntrs,200/scaling_factor)[0])
    
    def __find_centroid(self,cnt):
        cx = 0
        cy = 0
        tot_mass = sum([len(i) for i in cnt])
        for i in cnt:
            M = cv.moments(i)
            cx += len(i) * int(M['m10']/M['m00'])
            cy += len(i) * int(M['m01']/M['m00'])
        cx = int(cx/tot_mass)
        cy = int(cy/tot_mass)
        return (cx,cy)
    
    def find_holes(self, letter):
        bubble = self.bubble_dict[letter]
        #find holes
        point_list = [tuple(i[0][0]) for i in bubble[1:]]
        is_hole_list = []
        for point in point_list:
            point = (int(point[0]),int(point[1]))
            # print(point)
            is_hole_list.append(sum([max(0,cv.pointPolygonTest(cntr, point, False)) for cntr in bubble[1:]]) > 0)
        return is_hole_list
    
    # def get_metric(self,letter,cntr):
        #returns a double from 0 to 1, 
        #indicating area of "sketch" within bounds of bubble
        # bubble = self.bubble_dict[letter]
        # bubble = self.__scale_to_norm(bubble)
        # #normalize the letter & cntr
        # cntr_norm = self.get_norm_cnt(cntr)
        # cntr_norm = self.__scale_to_norm(cntr_norm)
        # to_ret += point_inside(point,cntr) if 
        # print(self.__find_holes(letter))
        
        
        #for every rotation, for every size, find maximum match 
           #for each segment of the bubble, check that there is a point lying inside, 
           #subtract points that are outside the contour
            


    # def get_prob(self,cnt, metric = "shape"):
    #     to_ret = {}
    #     for i in self.bubble_dict.keys():
    #         to_ret[i] = 0
    #     cnt_norm = self.scale_to_norm((cnt)) #put in a tuple
    #     for key,bubble in self.bubble_dict.items():
    #         if metric == "shape":
    #             bubble_norm = self.scale_to_norm(bubble)
    #             for i in bubble_norm:
    #                 to_ret[key] += cv.matchShapes(cnt_norm[0],i,1,0.0)
    #         elif metric == "area":
    #             to_ret[key] = cv.contourArea(bubble) - cv.contourArea(cnt_norm)
    #         else:
    #             return -1
    #     return to_ret
    
    def __y_sorted(self,cntrs):
        xtrm_point_list = {}
        for i in range(1, len(cntrs)):
            #xtrm_point_list[self.get_extreme(cntrs[i])] = cntrs[i]
            xtrm_point_list[self.__get_extreme(cntrs[i])] = i
        point_list = xtrm_point_list.keys()
        #x_sorted = sorted(point_list,key=lambda x: x[0])
        y_sorted = sorted(point_list,key=lambda y: y[1])
        sorted_dict = {i: xtrm_point_list[i] for i in y_sorted}
        return sorted_dict

class Lines:
    def __json_to_dict(self, file_name):
        with open(file_name) as json_file:
            lines_dict = json.load(json_file)
        to_ret = {}
        for key,val in lines_dict.items():
            for i in range(len(val)) :
                val[i] = (tuple(val[i][0]),tuple(val[i][1]))
            to_ret[key] = val
        return to_ret
    def __init__(self, json_file):
        self.letter_dict = self.__json_to_dict(json_file)
    
    def get_lines(self, char):
        return self.letter_dict[char]
    
    def __cart2pol(self, x, y):
            theta = np.arctan2(y, x)
            rho = np.hypot(x, y)
            return theta, rho
        
    def __pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def move_lines(self, lines,x,y):
        to_ret = []
        for pt1, pt2 in lines:
            pt1 = (int(pt1[0]+x),int(pt1[1]+y))
            pt2 = (int(pt2[0]+x),int(pt2[1]+y))
            to_ret.append((pt1, pt2))
        return to_ret

    def scale_lines(self, lines, scale):
        to_ret = []
        for pt1, pt2 in lines:
            pt1 = (int(pt1[0]*scale),int(pt1[1]*scale))
            pt2 = (int(pt2[0]*scale),int(pt2[1]*scale))
            to_ret.append((pt1, pt2))
        return to_ret
    
    def rotate_lines(self, bubble, lines, angle):
        to_ret = []
        for pt1, pt2 in lines:
            thetas, rhos = bubble.cart2pol(pt1[0], pt1[1])
            
            thetas = np.rad2deg(thetas)
            thetas = (thetas + angle) % 360
            thetas = np.deg2rad(thetas)

            xs, ys = bubble.pol2cart(thetas, rhos)
            pt1 = (xs,ys)
            
            thetas, rhos = bubble.cart2pol(pt2[0], pt2[1])
            
            thetas = np.rad2deg(thetas)
            thetas = (thetas + angle) % 360
            thetas = np.deg2rad(thetas)

            xs, ys = bubble.pol2cart(thetas, rhos)
            pt2 = (xs,ys)
            to_ret.append((pt1, pt2))
        return to_ret
    
    def draw_lines(self, lines, img, window_name = 'lines',copy = True):
        if copy:
            h,w = img.shape
            vis = np.ones((h, w), np.float32)
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        else: vis = img
        for pt1, pt2 in lines:
            cv.line(vis, pt1, pt2, (0,0,255), 1)
        cv.imshow(window_name,vis)
        cv.waitKey()
        return vis