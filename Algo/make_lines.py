import cv2 as cv
import numpy as np
import random
import json
import os
import re

from Utility import Bubble, Lines

def in_bubble(pt, cntr, holes):
        #inside the outer contour of the bubble, NOT inside the hole contour
        # print(pt)
        if cv.pointPolygonTest(cntr, pt , False) <= 0: return False
        for hole in holes:
            if cv.pointPolygonTest(hole, pt , False) > 0: return False
        else: return True

def create_intersection_lines(bubble, letter):
        #take a point on the contour (not a hole): take its two adjacent points
        #find the slope of these 2 adjacents
        #take the negative recipricol of the slope + make a step in both directions (from the original point)
        #check both steps to see if it is inside or outside (or in the hole) of the contour, one of the 2 should be valid
        #now continue to draw a "line" from the original point, in the direction of the valid step
        #draw until reaching another point on the edge of the contour or a point on the edge of the hole
        cntrs = bubble.bubble_dict[letter]
        cntrs = cntrs[1:]
        holes_list = bubble.find_holes(letter)
        boundry = []
        holes = []
        for i in range(len(holes_list)):
            if not holes_list[i]: boundry.append(cntrs[i])
            else: holes.append(cntrs[i])
        lines = []
        for cntr in boundry: 
            for point in range(len(cntr)):
                #slope = -dx/dy
                p1 = (point+1) % len(cntr)
                slope = -(cntr[p1][0][0] - cntr[point-1][0][0])/(cntr[p1][0][1] - cntr[point-1][0][1])
                vert_line = slope > 1000 or cntr[p1][0][1] == cntr[point-1][0][1]
                if vert_line: candidate = tuple([int(cntr[point][0][0]),int(cntr[point][0][1] + 1)])
                else: candidate = tuple([int(cntr[point][0][0] + 1),int(cntr[point][0][1] + slope)])#tuple (x,y)
                dir = 1 if cv.pointPolygonTest(cntr, candidate , False) > 0 else -1
                endpoint1 = (int(cntr[point][0][0]),int(cntr[point][0][1])) #tuple (x,y)
                endpoint2 = (int(endpoint1[0] + (0 if vert_line else dir)), 
                            int(endpoint1[1]+dir * (1 if vert_line else slope)))#tuple (x,y)
                while in_bubble(endpoint2,cntr,holes):
                    endpoint2 = (int(endpoint2[0] + (0 if vert_line else dir)), 
                            int(endpoint2[1]+dir * (1 if vert_line else slope)))#tuple (x,y)
                lines.append((endpoint1, endpoint2))
                #print((endpoint1, endpoint2))
        return lines

def make_json(folder_name, out_file_name, bubble):
    #example input: make_json("letter_templates", "bubbles.json")
    dictionary = {}
    for filename in os.listdir(folder_name):
        character = re.search("^.*\.", filename).group(0)[:-1]
        f = os.path.join(folder_name, filename)
        # checking if it is a file
        if os.path.isfile(f):
            norm_lines = create_intersection_lines(bubble, character)
            dictionary[character] = norm_lines
    # print(len(dictionary))
    with open(out_file_name, "w") as outfile:
        json.dump(dictionary, outfile)


def main():
    bubble = Bubble("bubbles.json")
    make_json("letter_templates", "lines.json",bubble)
    # lines_dict = json_to_dict("lines.json")
    # lines = Lines("lines.json")
    # print(lines_dict)
    # img = cv.imread("images/001.jpg",0)
    # i_line = lines.get_lines("i")
    # moved_i_line = lines.move_lines(i_line,500,500)
    # lines.draw_lines(moved_i_line, img)
    # img = cv.imread("images/001.jpg",0)
    # cnt = bubble.get_cnt('cap_b')
    # cnt_scaled = bubble.scale_cnt(cnt, 2)
    # cnt_rotated = bubble.rotate_cnt(cnt_scaled, 45)
    # cnt_moved = bubble.move_bubble(cnt_rotated,(500,500))
    # vis = bubble.draw_cnt(img, cnt_moved)
    # lines = create_intersection_lines(bubble, 'cap_b')
    # # print(lines)
    # # lines = scale_lines(lines, 2)
    # # lines = rotate_lines(bubble, lines, 45)
    # # lines = move_lines(lines,500,500)

     #create a dictionary where the key is a letter, and value is a list of pairs of endpoints that make up a line
main()