import cv2 as cv
import numpy as np
import random
import json

class Sketch:
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
        
    def scale_cnt(self,cnt, scale):
        # cx,cy,cnt_norm = move_contour_to_origin(cnt, return_shift=True)
        cnt_scaled = [0 for i in range(len(cnt))]
        for i in range(len(cnt)):
            cnt_scaled[i] = cnt[i] * scale
        return np.array(cnt_scaled)
        
    def __get_height_sketch_cnt(self, cnt):
        #helper for sort_cntrs (deals with sketches)
        transpose= np.transpose(cnt)
        height = np.amax(transpose, 2) - np.amin(transpose, 2)
        return height[1]
    
    def __max_height_sketch(self, cntrs):
        #helper for sort_cntrs (deals with sketches)
        # Calculate maximum rectangle height
        max_height = 0
        for i in range(1, len(cntrs)-2):#for weird paper lines, get rid of last 2
            h = self.__get_height_sketch_cnt(cntrs[i])
            # print(h)
            if h > max_height: max_height = h
        # print(max_height)
        return max_height
        
    def __y_sorted(self,cntrs):
        #helper for sort_cntrs (deals with sketches)
        xtrm_point_list = {}
        for i in range(1, len(cntrs)):
            #xtrm_point_list[self.get_extreme(cntrs[i])] = cntrs[i]
            xtrm_point_list[self.__get_extreme(cntrs[i])] = i
        point_list = xtrm_point_list.keys()
        #x_sorted = sorted(point_list,key=lambda x: x[0])
        y_sorted = sorted(point_list,key=lambda y: y[1])
        sorted_dict = {i: xtrm_point_list[i] for i in y_sorted}
        return sorted_dict
    
    def sort_cntrs(self, cntrs):
        #sorts the incoming sketch by line and from left to right 
        max_height = self.__max_height_sketch(cntrs)
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
            temp = []
            for point in x_sorted:
                temp.append(cntrs[by_y[point]])
            contours_sorted.append(temp)
        return contours_sorted, max_height
    
    def find_centroid(self,cnt): #for a single cntr
        tot_mass = sum([len(i) for i in cnt])
        cnt = cnt.astype(int)
        M = cv.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx,cy), tot_mass

    def __get_extreme(self,cnt):
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        return (extLeft[0],extTop[1])
    
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
    
    def combine_cntrs(self,cnt1:np.ndarray, cnt2:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
            Returns a single contour 

            Requires two contours
        """
        # print()
        pt_1 = cnt1[-1]
        pt_2 = cnt2[0]
        new_point = np.array([pt_1,pt_2])
        combined_cntr = np.concatenate((cnt1,cnt2),axis = 0)
        
        return combined_cntr, new_point
    
    def find_holes(self, line: list[np.ndarray] ) -> list[int]:
        """
            Returns a list of 0's and 1's indicating 1 if a hole contour found in the line
                and a list of the number of holes for each non hole cntr

            Requires a single line of the sketch with contours from left to right
        """
        point_list = [tuple(i[0][0]) for i in line[1:]] #a list of points, first point in each contour 
        is_hole_list = []
        for point in point_list:
            point = (int(point[0]),int(point[1]))
            is_hole_list.append(sum([max(0,cv.pointPolygonTest(cntr, point, False)) for cntr in line[1:]]) > 0)

        iter = 0
        holes_list = [0]
        while iter < len(is_hole_list):
            if is_hole_list[iter] == 1:
                holes_list[-1] += 1
            else:
                holes_list.append[0]
            
        return is_hole_list, holes_list
    
    def __remove_holes(self, line: list[np.ndarray]) -> list[np.ndarray]:
        """
            Returns a single line of the sketch with all holes removed maintaining order
                ex. (a's, o's, b's, g's, e's, etc.)
            
            Requires a single line of the sketch with contours from left to right

        """
        is_hole_list,holes_list = self.find_holes(line)
        for i in range(len(is_hole_list)):
            if is_hole_list[i] == 1 : del line[i]
        return line, holes_list
    
    def clean_sketch(self, sorted_sketch) -> list[np.ndarray]:
        """
            Returns list of individual contours in sketch by line from left to right
                    removing all holes (ex. a's, o's, e's, g's) and combining disconnected contours (ex. I)

            Requires sorted list of individual contours in sketch by line from left to right 
                *use self.sort_cntrs to reach this point            
        """
        holes_list_lines = []
        for i in range(1, len(sorted_sketch)):
            sorted_sketch[i],holes_list = self.__remove_holes(sorted_sketch[i])
            holes_list_lines.append(holes_list)
        
        return sorted_sketch,holes_list_lines
    
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
        self.sketch = Sketch()
        self.letter_match = {
            ".": "dot", 
            "A": "cap_a",
            "B": "cap_b",
            "C": "cap_c",
            "D": "cap_d",
            "E": "cap_e",
            "F": "cap_f",
            "G": "cap_g",
            "H": "cap_h",
            "I": "cap_i",
            "J": "cap_j",
            "K": "cap_k",
            "L": "cap_l",
            "M": "cap_m",
            "N": "cap_n",
            "O": "cap_o",
            "P": "cap_p",
            "Q": "cap_q",
            "R": "cap_r",
            "S": "cap_s",
            "T": "cap_t",
            "U": "cap_u",
            "V": "cap_s",
            "W": "cap_w",
            "X": "cap_x",
            "Y": "cap_y",
            "Z": "cap_z",
            "*": "astric",
            "?":"question_mark", 
            "\"":"quotation",
            ":": "colon"
        }
    
    def get_bubble(self,char):
        if char in self.letter_match:
            return self.bubble_dict[self.letter_match[char]]
        else:
            return self.bubble_dict[char]
        
    def move_bubble(self,cntrs,coord):
        cnt_moved = [0 for i in range(len(cntrs))]
        for i in range(len(cntrs)):
            cnt_moved[i] = (cntrs[i] + coord).astype(np.int32)
        cnt_moved = tuple(cnt_moved)
        return cnt_moved
        
    def __cart2pol(self, x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho
    
    def __pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    
    def draw_bubble(self,img,bub,index = -1, window_name = 'cnt', copy = True):
        if copy:
            h,w = img.shape
            vis = np.ones((h, w), np.float32)
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        else: 
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(vis,bub,index,(0,0,0),6)
        cv.imshow(window_name, vis)
        cv.waitKey()
        return vis

    # def draw_bubble(self,img,cnt,index = -1, window_name = 'cnt', copy = True):
        
    def draw_cnt_colored(self,img,cnt,window_name = 'cnt'):
        h,w = img.shape
        vis = np.ones((h, w), np.float32)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        for i in range(len(cnt)):
            cv.drawContours(vis,cnt,i,(0,0,0),6)
        cv.imshow(window_name, vis)
        cv.waitKey()
    
    def __get_bubble_size(self, cntrs):
        #intakes a "letter" from bubble.dict (the whole list of contours for that letter)
        #loop through each cnt, find the extremeTop, save the most extreme top
        #^^, extremeBottom, save the most extreme Bottom 
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
        norm = self.sketch.get_norm_cnt(cntrs)
        scaling_factor = self.__get_bubble_size(cntrs)
        # transpose = np.transpose(norm[0])
        # scaling_factor = np.min(np.amax(transpose,2) - np.amin(transpose,2))
        # print(scaling_factor)
        return tuple(self.sketch.scale_cnt(cntrs,200/scaling_factor)[0])
    
    def find_bubble_centroid(self, bub):
        cx = 0
        cy = 0
        tot_mass = sum([len(i) for i in bub])
        for i in bub:
            M = cv.moments(i)
            cx += len(i) * int(M['m10']/M['m00'])
            cy += len(i) * int(M['m01']/M['m00'])
        cx = int(cx/tot_mass)
        cy = int(cy/tot_mass)
        return (cx,cy), tot_mass
    
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
        self.letter_match = {
            ".": "dot", 
            "A": "cap_a",
            "B": "cap_b",
            "C": "cap_c",
            "D": "cap_d",
            "E": "cap_e",
            "F": "cap_f",
            "G": "cap_g",
            "H": "cap_h",
            "I": "cap_i",
            "J": "cap_j",
            "K": "cap_k",
            "L": "cap_l",
            "M": "cap_m",
            "N": "cap_n",
            "O": "cap_o",
            "P": "cap_p",
            "Q": "cap_q",
            "R": "cap_r",
            "S": "cap_s",
            "T": "cap_t",
            "U": "cap_u",
            "V": "cap_s",
            "W": "cap_w",
            "X": "cap_x",
            "Y": "cap_y",
            "Z": "cap_z",
            "*": "astric",
            "?":"question_mark", 
            "\"":"quotation",
            ":": "colon"
        }
    
    def get_lines(self, char):
        if char in self.letter_match:
            return self.letter_dict[self.letter_match[char]]
        else:
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
    
    def draw_lines(self, lines, img, window_name = 'lines',copy = True, color = (0,0,255)):
        if copy:
            h,w = img.shape
            vis = np.ones((h, w), np.float32)
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        else: vis = img
        for pt1, pt2 in lines:
            cv.line(vis, pt1, pt2, color, 1)
        cv.imshow(window_name,vis)
        cv.waitKey()
        return vis