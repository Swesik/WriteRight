from Utility import Bubble,Lines
import cv2 as cv
import numpy as np

def find_bubble_height(bubble_cntrs):
    low_point = np.array([float("inf"),float("inf")])
    high_point = np.array([float("-inf"),float("-inf")])
    for cnt in bubble_cntrs:
        transpose= np.transpose(cnt)
        new_low = np.amax(transpose, 2)
        low_point = new_low if new_low[1] < low_point[1] else low_point
        new_high = np.amin(transpose, 2)
        high_point = new_high if new_high[1] > high_point[1] else high_point
    # print(low_point)
    # print(high_point)
    return (low_point - high_point)[1][0]
        
def neighbhor_bubbles(bubble, scaled_bubbles):
    #pass a list of bubbles (should already be scaled to appropriate heights)
    #move bubbles next to each other
    moved_bubbles = [scaled_bubbles[0]]
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
        moved_bubbles.append(bubble.move_bubble(scaled_bubbles[i], (extRight[0]-extLeft[0],extBot1[1] - extBot2[1])))
    return moved_bubbles
def find_group_bubble_centroid(bubble, moved_bubbles):
    #pass a list of bubbles (should already be scaled to appropriate heights + moved next to each other)
    #find centroid of all bubbles
    total_mass = 0
    moment = (0,0)
    for bub in moved_bubbles:
        bub = [i.astype(int) for i in bub]
        (cx,cy), mass = bubble.find_bubble_centroid(bub)
        moment = ( moment[0] + cx * mass, moment[0] + cy*mass)
        total_mass += mass
    group_cm = (moment[0]/total_mass,moment[1]/total_mass)
    return group_cm

def match_group_by_height(bubble, handw_cntr, group, line_height):
    #find tallest bubble in group
    (hx,hy), hmass = bubble.find_centroid(handw_cntr)
    group_height = 0
    scaled_bubbles = []
    for letter in group:
        l_bubble = bubble.get_cnt(letter)
        h = find_bubble_height(l_bubble)
        scaled_bubbles.append(l_bubble)
        group_height = h if h > group_height else group_height
    scaling_factor = line_height/group_height
    scaled_bubbles = [bubble.scale_cnt(i, scaling_factor) for i in scaled_bubbles] 

    neighboring_bubbles = neighbhor_bubbles(bubble, scaled_bubbles)

    group_cm = find_group_bubble_centroid(bubble, neighboring_bubbles)
    moved_bubbles = []
    for i in range(len(neighboring_bubbles)):
        moved_bubbles.append(bubble.move_bubble(neighboring_bubbles[i], (hx-group_cm[0],hy-group_cm[1])))
    return moved_bubbles
    
def match_group_by_width(bubble, handw_cntr, group):
    #group is a string, can be 1 single letter or few letters
    x,y,sketch_width,h = cv.boundingRect(handw_cntr)#get size of handW_cntr (bounding box)
    (hx,hy), hmass = bubble.find_centroid(handw_cntr)
    # print(type(handw_cntr))
    sketch_cm, sketch_mass = bubble.find_centroid(handw_cntr)
    scaled_bubbles = []
    total_width = 0 
    for char in group: #get bubbles for each char in word
        char_cntr = bubble.get_cnt(char)
        scaled_bubbles.append(char_cntr)
        _,_,w,_= cv.boundingRect(char_cntr[1])
        total_width += w
    scaling_factor = sketch_width/total_width

    scaled_bubbles = [bubble.scale_cnt(i, scaling_factor) for i in scaled_bubbles]   #resize each bubble accordingly
    
    moved_bubbles = neighbhor_bubbles(scaled_bubbles)
    group_cm = find_group_bubble_centroid(moved_bubbles)
    #overlay bubbles
    for i in range(len(moved_bubbles)):
        moved_bubbles[i]=bubble.move_bubble(moved_bubbles[i], (hx-group_cm[0],hy-group_cm[1]))
    return moved_bubbles
    #overlay intersecting lines
     #+ match (intersecting lines)

    # overlay = bubble.move_cnt(moved_)
    #need to make combine cntrs, find itersection, 

def match(bubble, sorted_handw_cntrs, truth): 
    truth_words = truth.split(" ")
    for word in truth_words:
        bubble.get_cntr(char)
        line_height = bubble.max_height(sorted_handw_cntrs)
        
        sorted_handw_cntrs[0]

def main():
    bubble = Bubble("bubbles.json")
    lines = Lines("lines.json")
    img = cv.imread("images/002.jpg",0)
    # cv.imshow("window", img)
    # cv.waitKey()
    ret, thresh = cv.threshold(img, 127, 255, 0)
    handw_cntrs, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    sorted_handw_cntrs, line_height = bubble.sort_cntrs(handw_cntrs)
    
    truth = "I like apples \n My name is Sam \n I do not have dysgraphia".strip()
    print(truth)
    # bubble.draw_cnt(img,sorted_handw_cntrs,3)
    # (cx, cy), tot_mass = bubble.find_centroid(sorted_handw_cntrs[8])
    # cv.circle(img, (cx, cy), 7, (0,0,255), -1)
    # cv.imshow("window",img)
    # cv.waitKey()
    # print(bubble.find_centroid((sorted_handw_cntrs[2])))
    moved_bubbles = match_group_by_height(bubble,sorted_handw_cntrs[2][2], "like", line_height)
    # neighboring_bub 
    # print(len(moved_bubbles))
    moved_bubbles = [[i.astype(int) for i in moved] for moved in moved_bubbles]
    for moved in moved_bubbles:
        # for i in moved:
        #     print(len(i))        
        #     print(type(i))
        # moved = [i.astype(int) for i in moved]
        # moved = bubble.move_bubble(moved,(500,500))
        vis = bubble.draw_cnt(img,moved,index=1,copy=False)
        img = cv.cvtColor(vis,cv.COLOR_BGR2GRAY)
    # print(tmoved_bubbles)
    # cv.imshow("window", vis)
    # cv.waitKey()
    #vis = bubble.draw_cnt(img,moved_bubbles,copy=False)
    
    
main()
    