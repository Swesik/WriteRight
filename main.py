from Utility import Bubble,Lines, Sketch
import cv2 as cv
import numpy as np

class Matching:
    bubble_type = tuple[np.ndarray]
    lines_type = list[list[tuple[tuple,tuple]]]
    point_type = tuple[float,float]
    
    def __init__(self,bubbles_json_file: str ,lines_json_file: str):
        """
            Initializes a Matching objection.
            Requires name of bubbles.json file and lines.json file
        """
        self.bubble = Bubble(bubbles_json_file)
        self.lines = Lines(lines_json_file)
        self.sketch = Sketch()
    
    def find_bubble_height(self, bubble_cntrs: tuple[np.ndarray]) -> float:
        """
            Returns the height of the entire bubble 
            Requires a single character's orignal bubble (centered at origin, original size) 
                * can use Bubble.get_bubble(self, char)
        """
        low_point = np.array([float("inf"),float("inf")])
        high_point = np.array([float("-inf"),float("-inf")])
        for cnt in bubble_cntrs:
            transpose= np.transpose(cnt)
            new_low = np.amax(transpose, 2)
            low_point = new_low if new_low[1] < low_point[1] else low_point
            new_high = np.amin(transpose, 2)
            high_point = new_high if new_high[1] > high_point[1] else high_point
        return (low_point - high_point)[1][0]
            
    def __neighbhor_bubbles(self, scaled_bubbles: list[bubble_type], 
                            scaled_lines: lines_type) -> tuple[bubble_type, lines_type]:
        """
            Moves bubbles next to each other horizontally 
                right-most edges align with leftmost edges of previous bubble
            
            Returns a list of the moved bubbles and a list of the moved intersecting lines within the moved bubbles
            
            Requires a list of bubbles (should already be scaled to appropriate heights)
            and a list of intersecting lines within the list of bubbles
            
            scaled_bubbles/moved_bubbles type: a list of the scaled bubbles, each bubble is a list of contours which are nparrays
            scaled_lines/moved_lines type: a list of the intersecting lines for each bubble, each bubble has a list of lines, 
                               each line is a tuple of 2 endpoints (x,y)
        """  
        moved_bubbles = [scaled_bubbles[0]]
        moved_lines = [scaled_lines[0]]
        for i in range(1,len(scaled_bubbles)):
            extRight, extBot1, extBot2, extLeft = (float("-inf"),float("-inf")),(float("-inf"),float("-inf")),(float("-inf"),float("-inf")),(float("-inf"),float("-inf"))
            for cnt in moved_bubbles[i-1][1:]:
                right = tuple(cnt[cnt[:, :, 0].argmax()][0])
                bot1 = tuple(cnt[cnt[:, :, 1].argmax()][0])
                extRight = right if right[0] > extRight[0] else extRight
                extBot1 = bot1 if bot1[1] > extBot1[1] else extBot1
            for cnt in scaled_bubbles[i][1:]:
                left= tuple(cnt[cnt[:, :, 0].argmin()][0])
                bot2 = tuple(cnt[cnt[:, :, 1].argmax()][0])
                extLeft = left if left[0] > extLeft[0] else extLeft
                extBot2 = bot2 if bot2[1] > extBot2[1] else extBot2
            moved_bubbles.append(self.bubble.move_bubble(scaled_bubbles[i], (extRight[0]-extLeft[0],extBot1[1] - extBot2[1])))
            moved_lines.append(self.lines.move_lines(scaled_lines[i], extRight[0]-extLeft[0],extBot1[1] - extBot2[1]))
        return moved_bubbles, moved_lines
    
    def __find_group_bubble_centroid(self, moved_bubbles: list[bubble_type]) -> tuple[float,float]:
        """
            Returns the centroid of a group of adjacent bubbles

            Requires a list of bubbles (should already be scaled to appropriate heights + moved next to each other)
                *use neighbhor_bubbles to reach this step
            
            moved_bubbles type: a list of the scaled adjacent bubbles, each bubble is a list of contours which are nparrays
        """
        total_mass = 0
        moment = (0,0)
        for bub in moved_bubbles:
            bub = [i.astype(int) for i in bub]
            (cx,cy), mass = self.bubble.find_bubble_centroid(bub)
            moment = ( moment[0] + cx * mass, moment[0] + cy*mass)
            total_mass += mass
        group_cm = (moment[0]/total_mass,moment[1]/total_mass)
        return group_cm

    def match_group_by_height(self, handw_cntr: list[np.ndarray], group: str, line_height: float) -> tuple[list[bubble_type],lines_type]:
        """
            Returns a list of moved bubbles and moved intersecting lines which should directly overlay onto the sketch 
            
            moved_bubbles type: a list of the scaled adjacent bubbles, each bubble is a list of contours which are nparrays
            
            moved_lines type: a list of the intersecting lines for each bubble, each bubble has a list of lines, 
                              each line is a tuple of 2 endpoints (x,y)
                              
            Requires a sketch object to match to, a string of letters (group) which we "guess" are represented by the sketch, 
            and a double line_height which is a general estimate of the height of a character in the sketch

        """
        (hx,hy), hmass = self.sketch.find_centroid(handw_cntr)
        group_height = 0 #find tallest bubble in group
        scaled_bubbles = []
        scaled_lines = []
        for letter in group:
            l_bubble = self.bubble.get_bubble(letter)
            h = self.find_bubble_height(l_bubble)
            scaled_bubbles.append(l_bubble)
            scaled_lines.append(self.lines.get_lines(letter))
            group_height = h if h > group_height else group_height
        scaling_factor = line_height/group_height #use the tallest bubble to determine scaling factor for all bubbles
        scaled_bubbles = [[self.sketch.scale_cnt(j, scaling_factor) for j in i] for i in scaled_bubbles] 
        scaled_lines = [self.lines.scale_lines(i, scaling_factor)  for i in scaled_lines]
        
        neighboring_bubbles, neighboring_lines = self.__neighbhor_bubbles(scaled_bubbles, scaled_lines)

        group_cm = self.__find_group_bubble_centroid(neighboring_bubbles)
        moved_bubbles = []
        moved_lines = []
        for i in range(len(neighboring_bubbles)):
            moved_bubbles.append(self.bubble.move_bubble(neighboring_bubbles[i], (hx-group_cm[0],hy-group_cm[1])))
            moved_lines.append(self.lines.move_lines(neighboring_lines[i], hx-group_cm[0],hy-group_cm[1]))
        return moved_bubbles, moved_lines
    
    def draw_match(self,img: np.ndarray,moved_bubbles: list[bubble_type],moved_lines: lines_type, color = (0,0,255)):
        """
            Draws overlaying bubbles and intersecting lines on a given handwritten sketch image
            
            Returns none

            Requires an img containing a handwritten sketch as well as a list of moved bubbles and a list of moved lines 
                *use match_by_group_height() to reach this point
            
        """
        for moved in moved_bubbles:
            vis = self.bubble.draw_bubble(img,moved,index=-1,copy=False)
            img = cv.cvtColor(vis,cv.COLOR_BGR2GRAY)
        img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        for line in moved_lines:
            vis = self.lines.draw_lines(line, img, copy = False, color=color)
    
    def __is_intersected(self,pt_1: point_type,pt_2: point_type, line:tuple[point_type,point_type]) -> bool:
        """
            Returns false if intersection is invalid

            Requires 2 consecutive points on the sketch contour and a line from the intersecting lines list for the corresponding bubble
        """
        ccw = lambda A,B,C : (C[1] - A[1]) * (B[0]-A[0]) > (B[1]-A[1])  * (C[0] - A[0])
        return ccw(pt_1, line[0], line[1]) != ccw(pt_2, line[0], line[1]) and ccw(pt_1, pt_2, line[0]) != ccw(pt_1, pt_2, line[1])
        
    def find_intersections(self,moved_lines: lines_type, sketch_cntr: np.ndarray) -> set:
        """
            Returns a subset of the intersecting lines which have been successfully intersected 
            
            Requires a list of moved bubbles and moved lines which directly overlay the sketch_cntr
                *use match_by_group_height() to reach this point

        """
        intersected_lines = set()
        for letter_lines in moved_lines:
            for lines in letter_lines:
                    for i in range(1,len(sketch_cntr)):
                        pt_1 = sketch_cntr[i-1]
                        pt_2 = sketch_cntr[i]
                        if self.__is_intersected(pt_1[0],pt_2[0],lines):
                            intersected_lines.add(lines)
        return intersected_lines
        
    def __find_best_match(self,curr_word:str, handw_cntr: np.ndarray,line_height: float) -> tuple[str,lines_type]:
        """
            Return string of best_match within the word
            
            Requires substring of unmatched letters in current word and the single contour from the sketch being matched
            and the estimated line height for sizing 
            
        """
        best_score = 0 
        best_match = curr_word[0]
        possible_letters = ""
        best_intersected_lines = []
        for letter in curr_word: 
            possible_letters += letter
            moved_bubbles, moved_lines = self.match_group_by_height(handw_cntr, possible_letters, line_height) 
            # moved_bubbles = [[i.astype(int) for i in moved] for moved in moved_bubbles]
            # self.draw_match(handw_img, moved_bubbles,moved_lines)

            intersected_lines = self.find_intersections(moved_lines,handw_cntr)
            # self.draw_match(handw_img, moved_bubbles, [intersected_lines], (0,255, 0))

            tot_lines_len = sum([len(moved_lines[i]) for i in range(len(moved_lines))])
            curr_score = len(intersected_lines)/tot_lines_len
            if curr_score > best_score:
                best_score = curr_score
                best_match = possible_letters
                best_intersected_lines = intersected_lines
        return best_match, best_intersected_lines

    def match(self, handw_img: cv.Mat, truth: str):
        """
            Runs matching sequence, reads in an image, sorts sketch contours and moved and draws overlaying bubbles and intersecting lines
            
            Returns moved bubbles and moved lines and intersected lines for fully matched sketch

            Requires an img with the full handwritten sketch, a string with the ground-truth
            
            for each letter in each word tries matching to next cntr, if fails add next letter and match together, 



            TODO: vertically oriented contours in the sketch (aka holes in a or disconnected I's) these should be treated as 1 
                  combined contour. They are in the same line (by sorted_handw_cntrs and have similar centroids/ horizontal placement within the line)
                  match the contours as one combined contour
        """
        
        # cv.imshow("og",handw_img)
        ret, thresh = cv.threshold(handw_img, 127, 255, 0)
        handw_cntrs, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #finds all contours in sketch img
        
        sorted_handw_cntrs, line_height = self.sketch.sort_cntrs(handw_cntrs) #sorts contours by line from left to right
        
        temp_img = 255*np.ones_like(handw_img)
        for i in range(len(sorted_handw_cntrs[3])):
            temp_img = 255*np.ones_like(temp_img)
            cv.drawContours(temp_img,sorted_handw_cntrs[3],i,(0,0,255),1)
            cv.imshow("window",temp_img)
            cv.waitKey()
    
        best_match_list = []
        intersected_lines_list = []
        
        truth_words = truth.split()
        word_iter = 0
        curr_word = truth_words[0]
        
        temp_img = 255*np.ones_like(handw_img)
        for line in sorted_handw_cntrs[1:]:
            # print(sorted_handw_cntrs[1].shape)
            for cntr in line:
                temp_img = 255*np.ones_like(temp_img)
                cv.drawContours(temp_img,cntr,-1,(0,0,255),1)
                cv.imshow("window",temp_img)
                cv.waitKey()
                # print(cntr.shape)
                best_match, intersected_lines = self.__find_best_match(curr_word,cntr,line_height)
                print(best_match)
                best_match_list.append(best_match)
                intersected_lines_list.append(intersected_lines)
                if len(best_match) < len(curr_word):
                    curr_word = curr_word[len(best_match):]
                else:
                    word_iter += 1
                    curr_word = truth_words[word_iter]           
        

def main():
    img = cv.imread("images/002.jpg",0)
    truth = "I like pears \n My name is Sam \n I do not have dysgraphia".strip()
    print(truth)
    
    matcher = Matching("bubbles.json","lines.json")
    matcher.match(img,truth)
    
    
    
main()
    