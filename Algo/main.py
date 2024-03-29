from Utility import Bubble,Lines, Sketch
from queue import PriorityQueue
import cv2 as cv
import numpy as np
import copy
import time
class Matching:
    bubble_type = tuple[np.ndarray]
    lines_type = list[list[tuple[tuple,tuple]]]
    point_type = tuple[float,float]
    state_type = tuple[float,list[float],list[np.ndarray],list[int],list[str],lines_type]
    
    def __init__(self,bubbles_json_file: str ,lines_json_file: str):
        """
            Initializes a Matching objection.
            Requires name of bubbles.json file and lines.json file
        """
        self.bubble = Bubble(bubbles_json_file)
        self.lines = Lines(lines_json_file)
        self.sketch = Sketch()
        self.line_height = 0
    
    def find_bubble_height(self, bubble_cntrs: tuple[np.ndarray]) -> float:
        """
            Returns the height of the entire bubble 
            Requires a single character's orignal bubble (centered at origin, original size) 
                * can use Bubble.get_bubble(self, char)
        """
        low_point = np.array([float("-inf"),float("-inf")])
        high_point = np.array([float("inf"),float("inf")])
        for cnt in bubble_cntrs[1:]:
            transpose= np.transpose(cnt)
            new_low = np.amax(transpose, 2)
            low_point = new_low if new_low[1] > low_point[1] else low_point
            new_high = np.amin(transpose, 2)
            high_point = new_high if new_high[1] < high_point[1] else high_point
        return (low_point - high_point)[1][0]
            
    def __neighbhor_bubbles(self, scaled_bubbles: list[bubble_type], 
                            scaled_lines: lines_type) -> tuple[bubble_type, lines_type]:
        """
            Moves bubbles next to each other horizontally 
                left-most edges align with right-most edges of previous bubble
            
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
            moment = ( moment[0] + cx * mass, moment[1] + cy*mass)
            total_mass += mass
        group_cm = (moment[0]/total_mass,moment[1]/total_mass)
        return group_cm

    def match_group_by_height(self, handw_cntr: list[np.ndarray], group: str) -> tuple[list[bubble_type],lines_type]:
        """
            Returns a list of moved bubbles and moved intersecting lines which should directly overlay onto the sketch 
            
            moved_bubbles type: a list of the scaled adjacent bubbles, each bubble is a list of contours which are nparrays
            
            moved_lines type: a list of the intersecting lines for each bubble, each bubble has a list of lines, 
                              each line is a tuple of 2 endpoints (x,y)
                              
            Requires a sketch object to match to, a string of letters (group) which we "guess" are represented by the sketch, 
            and a double line_height which is a general estimate of the height of a character in the sketch (part of the matching class, does not need to be passed in )

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
        scaling_factor = self.line_height/group_height #use the tallest bubble to determine scaling factor for all bubbles
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
    
    def __is_intersected(self,pt_1: point_type,pt_2: point_type, line:tuple[point_type,point_type], threshold:float = 20) -> bool:
        """
            Returns false if intersection is invalid
                for an intersection to be valid the 2 lines must ALSO be close to perpendicular to each other (some degree of error allowed)
            
            Requires 2 consecutive points on the sketch contour and a line from the intersecting lines list for the corresponding bubble
            
        """
        sketch_angle = np.rad2deg(np.arctan2(pt_1[1] - pt_2[1])/(pt_1[0]-pt_2[0]))
        parallel_angle= np.rad2deg(np.arctan2(-(line[0][0]-line[1][0])/(line[0][1] - line[1][1])))
        ccw = lambda A,B,C : (C[1] - A[1]) * (B[0]-A[0]) > (B[1]-A[1])  * (C[0] - A[0])
        return ccw(pt_1, line[0], line[1]) != ccw(pt_2, line[0], line[1]) and ccw(pt_1, pt_2, line[0]) != ccw(pt_1, pt_2, line[1]) \
            and (sketch_angle < (parallel_angle + threshold) % 360 or sketch_angle > (parallel_angle - threshold)% 360)
        
    def find_intersections(self,moved_lines: list[lines_type], sketch_cntr: np.ndarray, ignored_lines:list[lines_type] = []) -> set:
        """
            TODO: fix this!!!
                idea-- iterate through each line segment of the sketch (hopefully in order of drawing),
                    then iterate through each intersecting line (also in order, beginning at the end of the last segments intersected line)
                    stop iterating when the segment ends and move on to the next segment 
                ?? are the line segments ordered the way we want??? clockwise?? from top to bottom?? 
                alternately we can order the lines ourselves in 2 sets (left to right and up to down)
                TODO: find the starting point of the sketch [0] and locate the closest points in the closed bubble to this point
                      then start from these 2 locations on the bubble and travel in opposite directions (parallel) around the bubble 
            Returns a subset of the intersecting lines which have been successfully intersected 
            
            Requires a list of moved bubbles and moved lines which directly overlay the sketch_cntr
                requires a list of connecting lines which are not part of the original sketch and therefore should not be matched (ignored_lines)
                *use match_by_group_height() to reach this point

        """
        # t_line = 0#index of current intersecting line from the top
        # b_line = 0#index of current intersecting line from the bottom 
        intersected_lines = set()
        
        # #find the line which is closest to the sketch start
        # sketch_start = sketch_cntr[0]
        # lines_start = 0
        # for letter_lines in moved_lines:
        #     for lines in letter_lines:
        #         if lines not in ignored_lines:
                    
                        
                        
                        
            
        # for i in range(1,len(sketch_cntr)): #loop through every line segment in the sketch 
        #     pt_1 = np.transpose(sketch_cntr[i-1]) 
        #     pt_2 = np.transpose(sketch_cntr[i])
        #     print("p1 = ", pt_1)
        #     print("p1 = ", [pt_1[0],pt_1[1]])
        #     print("p2 = ", pt_2)
        #     print("p2 = ", [pt_2[0],pt_2[1]])
        #     midpoint = np.array([[(pt_1[0][0] + pt_2[0][0])/2],[(pt_1[1][0] + pt_2[1][0])/2]]) 
        #     pt_1_trans = pt_1 - midpoint
        #     pt_2_trans = pt_2 - midpoint
        #     slope = (pt_2[1][0] - pt_1[1][0])/(pt_2[0][0] - pt_1[0][0]) #calculate slop of this line segment
        #     theta = np.arctan(slope)
        #     c, s = np.cos(theta), np.sin(theta)
        #     rot_matrix = np.array(((c, -s), (s, c)))#calculate the transformation matrix 
        #     pt_1_trans = np.matmul(rot_matrix,pt_1_trans) + midpoint
        #     pt_2_trans = np.matmul(rot_matrix,pt_2_trans) + midpoint
            
        #     for i in 
            
            
            
            #check next intersecting line (__is_intersected)
            #if __is_insersected returns false
                #then transform the intersecting line and check if it is within the range of the line segment
            
        #-----------------------------------------old version-----------------------------------------------#
        for letter_lines in moved_lines:
            for lines in letter_lines:
                if lines not in ignored_lines:
                    for i in range(1,len(sketch_cntr)):
                        pt_1 = sketch_cntr[i-1]
                        pt_2 = sketch_cntr[i]
                        if self.__is_intersected(pt_1[0],pt_2[0],lines):
                            intersected_lines.add(lines)
        return intersected_lines
        
    def __find_best_match(self,curr_word:str, handw_cntr: np.ndarray) -> tuple[str,lines_type]:
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
            moved_bubbles, moved_lines = self.match_group_by_height(handw_cntr, possible_letters, self.line_height) 
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
        print(best_match, best_score)
        return best_match, best_intersected_lines
    
    def __neighbhor_cntrs(self, cntrs_list: list[np.ndarray]) -> list[np.ndarray]:
        """
            Moves contours next to each other horizontally 
                left-most edges align with right-most edges of previous contour
            
            Returns a list of the moved contours
            
            Requires a list of contours 
        """  
        moved_contours = [cntrs_list[0]]
        for i in range(1,len(cntrs_list)):
            extRight, extBot1, extBot2, extLeft = (float("-inf"),float("-inf")),(float("-inf"),float("-inf")),(float("-inf"),float("-inf")),(float("-inf"),float("-inf"))
            cnt = moved_contours[i-1]
            right = tuple(cnt[cnt[:, :, 0].argmax()][0])
            bot1 = tuple(cnt[cnt[:, :, 1].argmax()][0])
            extRight = right if right[0] > extRight[0] else extRight
            extBot1 = bot1 if bot1[1] > extBot1[1] else extBot1
            cnt = cntrs_list[i]
            left = tuple(cnt[cnt[:, :, 0].argmin()][0])
            bot2 = tuple(cnt[cnt[:, :, 1].argmax()][0])
            extLeft = left if left[0] > extLeft[0] else extLeft
            extBot2 = bot2 if bot2[1] > extBot2[1] else extBot2
            # print(cntrs_list[i])
            coord = (extRight[0]-extLeft[0],extBot1[1] - extBot2[1])
            moved_cntr = (cntrs_list[i] + coord).astype(np.int32)
            moved_contours.append(moved_cntr)
            # print(cntrs_list[i])
        return moved_contours
    
    def score_heuristic(self,remaining_truth:str, remaining_cntrs: list[np.ndarray], img:np.ndarray):
        """
            TODO: use naive segmentation for scoring the bubbles
            
            Returns a float score of the estimated match between the remaining unmatched letters and unmatched contours

            Requires a string of remaining unmatched letters and contours
                and a list of segments 
        """
        moved_cntrs = self.__neighbhor_cntrs(copy.deepcopy(remaining_cntrs)) #move all remaining cntrs together 
        cntr = moved_cntrs[0]
        ignore_lines = []
        for i in range(1, len(moved_cntrs)):
            cntr, new_point = self.sketch.combine_cntrs(cntr, moved_cntrs[i]) 
            ignore_lines.append(new_point)
        moved_bubbles, moved_lines = self.match_group_by_height(cntr, remaining_truth)
        
        
        #for each line segment in sketch
            #for each chunk
                #check if endpoints of the line segment are IN the chunk
                #check slope of chunks
        
        self.draw_match(img,moved_bubbles,moved_lines)
        return 0 
    
    def score_match(self, letters:str, cntr: np.ndarray):
        """
            Returns a float score of the match between a grouping of letters and a contour 
            
            Requires a string of adjacent letters and a single contour (can be previously combined)

        """
        moved_bubbles, moved_lines = self.match_group_by_height(cntr, letters, self.line_height) 
        # moved_bubbles = [[i.astype(int) for i in moved] for moved in moved_bubbles]
        # self.draw_match(handw_img, moved_bubbles,moved_lines)

        intersected_lines = self.find_intersections(moved_lines, cntr)
        # self.draw_match(handw_img, moved_bubbles, [intersected_lines], (0,255, 0))

        tot_lines_len = sum([len(moved_lines[i]) for i in range(len(moved_lines))])
        curr_score = len(intersected_lines)/tot_lines_len
        return curr_score
    
    def next_states(self, parent_state: state_type, ground_truth: str) -> tuple[state_type,state_type,state_type]:
        """
            Returns 3 states: 
                a. continue to next match: moves on to the next letter and next cntr
                b. combine contours: stays on the current letter and combines current cntr with next cntr 
                c. add groupings: appends the next letter to the current letter grouping and stays on the current cntr
            
            Requires the parent state in the form: 
            A state is a tuple containing:
                0: total cost of state (current cost + heurisitc),
                1: cost list (list of all previous edge costs)
                2: its own modified list of contours,
                3: list of holes,
                4: list of strings (letter groupings) for currently matched contours ,
                5: list of ignored lines in intersecting lines (connecting 2 contours)
            and a string which represents the ground_truth of the entire line (with no spaces)
        """
        
        #find next letter to use in the ground_truth 
        index_next_letter = sum([len(i) for i in parent_state[4]])
        next_letter = ground_truth[index_next_letter]
        
        #next: moving on to the next contour 
        next_score_move = score_match(next_letter, parent_state[2][len(parent_state[1])])
        next_heuristic  = score_heuristic(ground_truth[index_next_letter+1:], parent_state[2][len(parent_state[1])+1:])
        next_state = (next_heuristic + next_score_move + sum(parent_state[1]), parent_state[1] + next_score_move, parent_state[2], parent_state[3],parent_state[4] + next_letter,parent_state[5])
       
        #combine: combining 2 contours 
        combine_cntr, connection_line = self.sketch.combine_cntrs(parent_state[2][len(parent_state[1])], parent_state[2][len(parent_state[1])+1])
        combine_cntr_list = copy.deepcopy(parent_state[2]) #copy the parent's list of contours
        combine_cntr_list[len(parent_state[1])-1] = combine_cntr #replace with the modified contour and delete the combined 2
        combine_cntr_list.remove(len(parent_state[1]))
        combine_move_score = score_match(parent_state[4], combine_cntr) #score the move 
        combine_cost_list = parent_state[1] #copy the parent's list of costs
        combine_cost_list[-1] = combine_move_score #modify the latest cost to represent this move
        combine_total_cost = sum(combine_cost_list) + score_heuristic(ground_truth[index_next_letter:], combine_cntr_list[len(combine_cost_list)+1:]) #calculate the total cost with the heuristic
        combine_ignore_lines = parent_state[5].append(connection_line) #add the connecting line to the parent's list of ignored lines
        combine_state = (combine_total_cost, combine_cost_list, combine_cntr_list, parent_state[3],parent_state[4],)
        
        #add: adding letter to grouping 
        add_letter_list = copy.deepcopy(parent_state[4]) #copy parent's list of letter groupings
        add_letter_list[-1] = parent_state[4][-1]+next_letter #replace last letter grouping with modified (add next letter)
        add_score_move = score_match(add_letter_list[-1], parent_state[2][len(parent_state[1])])
        add_cost_list = parent_state[1] #copy the parent's list of costs
        add_cost_list[-1] = combine_move_score #modify the latest cost to represent this move
        add_total_cost = sum(combine_cost_list) + score_heuristic(ground_truth[index_next_letter+1:],parent_state[2][len(parent_state[1])+1:]) #calculate the total cost with the heuristic
        add_state = (add_total_cost, add_cost_list, parent_state[2], parent_state[3], add_letter_list, parent_state[5])
        return (next_state,combine_state,add_state)
        
    def a_star(self, ground_truth:str, cntr_list: list[np.ndarray]):
        """
            Returns the best matching state
            A state is a tuple containing:
                total cost of state (current cost + heurisitc),
                cost list (list of all previous edge costs),
                its own modified list of contours,
                list of holes,
                list of strings (letter groupings) for currently matched contours ,
                list of ignored lines in intersecting lines (connecting 2 contours)
            Requires ground_truth string 
        """
        
        pq = PriorityQueue()
        init_state = (10000,[score_match(ground_truth[0], cntr_list[0])],cntr_list,self.sketch.find_holes(cntr_list),[ground_truth[0]],[])
        pq.put(init_state)
        
        while pq:
            
            curr_node = pq.get()#pop top item from pq
            
            if is_terminal(curr_node): return curr_node #check if curr_node is goal

            for state in next_states(copy.deepcopy(curr_node), ground_truth): #push children onto the pq
                if state: pq.put(state)
            
        
              
    def match_line(self, truth:str):
        """
            Requires a str which is a single line of the ground truth 

            first clean sketch (removes holes + keep track of holes)

            then combine contours and match with a single character 

        """
    def match(self, handw_img: cv.Mat, truth: str):
        """
            Runs matching sequence, reads in an image, sorts sketch contours and moved and draws overlaying bubbles and intersecting lines
            
            Returns moved bubbles and moved lines and intersected lines for fully matched sketch

            Requires an img with the full handwritten sketch, a string with the ground-truth
            
            for each letter in each word tries matching to next cntr, if fails add next letter and match together, 

            TODO: vertically oriented contours in the sketch (aka holes in a or disconnected I's) these should be treated as 1 
                  TODO: clean sketch of holes, remove the hole contours from the list and indicate how many holes a contour has (in a list)
                        end result [2, 0 , 0 , 1, 1, 0 , 1] for the sketch "Bubble"
                  TODO: Make Huristic
                  
                  combined contour. They are in the same line (by sorted_handw_cntrs and have similar centroids/ horizontal placement within the line)
                  match the contours as one combined contour

                  backtracking approach: at each step choose to...
                  1. match a whole word to a possibly combined contour
                  2. combine 2 contours
                  3. go back (scored poorly)

                  if a contour is significantly small (as compared to line height) it SHOULD be combined 

                  using remove_holes to identify where there are holes and get a better idea of what character is there and should match??
                  
                  RL approach: find the greatest total intersected lines score for 1 line of the ground truth 
                  options: 
                    1. combine contours
                    2. match 
                    3. combine bubbles + match
                    
                  Terminal state - all contours and all bubbles are matched
                  
                  Action states - match curr_cntr to curr_group, iterate to next group, combine cntr
        """
        
        #find the cntrs from handwriting 
        # cv.imshow("og",handw_img)
        ret, thresh = cv.threshold(handw_img, 127, 255, 0)
        handw_cntrs, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #finds all contours in sketch img
        
        sorted_handw_cntrs, self.line_height = self.sketch.sort_cntrs(handw_cntrs) #sorts contours by line from left to right
        temp_img = 255*np.ones_like(handw_img)
        cv.drawContours(temp_img, sorted_handw_cntrs[3], -1, (0,0,255),1)
        cv.imshow("window",temp_img)
        cv.waitKey()
        
        temp_img = 255*np.ones_like(handw_img)
        moved_cntrs = self.__neighbhor_cntrs(sorted_handw_cntrs[3])
        # print(len(sorted_handw_cntrs[3]))
        
        temp_img = 255*np.ones_like(temp_img)
        cv.drawContours(temp_img, moved_cntrs, -1, (0,0,255),1)
        cv.imshow("window2",temp_img)
        cv.waitKey()
        
        temp_img = 255*np.ones_like(temp_img)
        self.score_heuristic("MynameisEthan",sorted_handw_cntrs[3],temp_img)
        
        # cv.drawContours(temp_img,handw_cntrs,0,(0,0,255),1)
        # cv.imshow("window",temp_img)
        # cv.waitKey()
    
        # for i in range(len(sorted_handw_cntrs[3])):
        #     temp_img = 255*np.ones_like(temp_img)
        #     cv.drawContours(temp_img,sorted_handw_cntrs[3],i,(0,0,255),1)
        #     cv.imshow("window",temp_img)
        #     cv.waitKey()

        # cv.drawContours(temp_img, sorted_handw_cntrs[3], 7, (0,0,255),1)
        # cv.imshow("window",temp_img)
        # cv.waitKey()
        
        # cv.drawContours(temp_img, sorted_handw_cntrs[3], 8, (0,0,255),1)
        # cv.imshow("window",temp_img)
        # cv.waitKey()
        # print(len(sorted_handw_cntrs[2]))
        # combined_cntr, connection_point = self.sketch.combine_vert_cntrs(sorted_handw_cntrs[3][7],sorted_handw_cntrs[3][8])
        # sorted_handw_cntrs[3].append(combined_cntr)
        # temp_img2 = 255*np.ones_like(temp_img)
        # cv.drawContours(temp_img2, sorted_handw_cntrs[3], len(sorted_handw_cntrs[3])-1, (0,0,255),1)
        # cv.imshow("window2",temp_img2)
        # cv.waitKey()
        # best_match_list = []

        # intersected_lines_list = []
        
        # truth_words = truth.split()
        # word_iter = 0
        # curr_word = truth_words[0]
        
        # temp_img = 255*np.ones_like(handw_img)
        # for line in sorted_handw_cntrs[1:]:
        #     # print(sorted_handw_cntrs[1].shape)
        #     for cntr in line:
        #         # temp_img = 255*np.ones_like(temp_img)
        #         # cv.drawContours(temp_img,cntr,-1,(0,0,255),1)
        #         # cv.imshow("window",temp_img)
        #         # cv.waitKey()
        #         # print(cntr.shape)
        #         best_match, intersected_lines = self.__find_best_match(curr_word,cntr,line_height)
        #         #print(best_match)
        #         best_match_list.append(best_match)
        #         intersected_lines_list.append(intersected_lines)
        #         if len(best_match) < len(curr_word):
        #             curr_word = curr_word[len(best_match):]
        #         else:
        #             word_iter += 1
        #             curr_word = truth_words[word_iter]           
        

def main():
    img = cv.imread("images/002.jpg",0)
    truth = "I like pears \n My name is Sam \n I do not have dysgraphia".strip()
    print(truth)
    
    matcher = Matching("jsons/bubbles.json","lines.json")
    matcher.match(img,truth)
    
    
    
main()
    