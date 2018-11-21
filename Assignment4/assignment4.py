import numpy as np
import cv2
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain

# 1. Image thresholding
image= cv2.imread("image1.jpg",1)
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold(image,70,255,cv2.THRESH_BINARY)
cv2.imwrite("image_filtered.png", img)

   
# 2. Calculating area in quadtrees
size = 0
tree = []
size = image.size[0] * image.size[1]
for i in range(0, self.size):
    tree.append(Pixel()) 
count = 0
for i in range(image.size[0] - 1, 0, -2): 
    for j in range(image.size[1] - 1, 0, -2): 
        tree[size - 1 - 4 * count] = Pixel(img[i, j],  
                Point(i, j),  
                Point(i, j)) 
        tree[size - 2 - 4 * count] = Pixel(img[i, j - 1],  
                Point(i, j - 1), 
                Point(i, j - 1)) 
        tree[size - 3 - 4 * count] = Pixel(img[i - 1, j],  
                Point(i - 1, j),  
                Point(i - 1, j)) 
        tree[size - 4 - 4 * count] = Pixel(img[i - 1, j - 1],  
                Point(i - 1, j - 1),  
                Point(i - 1, j - 1)) 
        count += 1
        
for i in range(size - 4 * count - 1, -1, -1): 
    tree[i] = Pixel( 
        [(tree[4 * i + 1].R + tree[4 * i + 2].R + tree[4 * i + 3].R + tree[4 * i + 4].R) / 4, 
         (tree[4 * i + 1].G + tree[4 * i + 2].G + tree[4 * i + 3].G + tree[4 * i + 4].G) / 4, 
         (tree[4 * i + 1].B + tree[4 * i + 2].B + tree[4 * i + 3].B + tree[4 * i + 4].B) / 4], 
        tree[4 * i + 1].topLeft, 
        tree[4 * i + 4].bottomRight)
print(count)    
#3. Freeman 4-connectivity chain code representation 
start_point = (0, 0)
for i, row in enumerate(img):
    for j, value in enumerate(row):
        try:
            #if value == 255 and img[i][j+1]==0:
                #start_point = (i, j)
            if value == 0:
                start_point=(i,j-1)
                break
            else:
                continue           
        except:
            continue
    if start_point !=(0,0):
        break
print(start_point)
directions = [ 0,  1,  2, 3]
dir2idx = dict(zip(directions, range(len(directions))))

change_i =   [ 1, 0,-1,0] # x or columns

change_j =   [ 0, 1, 0,-1] # y or rows
border = []
chain = []
curr_point = start_point
for direction in directions:
    idx = dir2idx[direction]
    new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
    if img[new_point] != 0: # if is ROI
        border.append(new_point)
        chain.append(direction)
        curr_point = new_point       
        break
count = 0
while curr_point != start_point:
    #figure direction to start search
    dirs=[]
    for i in (0,1,3):
        dirs.append((direction+i)%4)
    for direction in dirs:
        idx = dir2idx[direction]
        new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
        try:
            if img[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        except:
            continue
    if count == 1000: break
    count += 1
print(count)
print(chain)

# 4. Convex Hull
im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hull = []
 
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))
# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)
