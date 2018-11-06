import numpy as np
import cv2
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def mshift(K):
    row=K.shape[0]
col=K.shape[1]

J= row * col
Size = row,col,3
R = np.zeros(Size, dtype= np.uint8)
D=np.zeros((J,5))
arr=np.array((1,3))
print(len(D))

#cv2.imshow("image", K)

counter=0  
iter=1.0        

threshold=30
current_mean_random = True
current_mean_arr = np.zeros((1,5))
below_threshold_arr=[]

# 이미지를 feature space D로 변환
for i in range(row):
    for j in range(col):
        arr= K[i][j]        
        for k in range(5):
            if(k>=0) and (k <=2):
                D[counter][k]=arr[k]
            else:
                if(k==3):
                    D[counter][k]=i
                else:
                    D[counter][k]=j
        counter+=1
while(len(D) > 0):
    print(len(D))
#feature space에서의 특정 row를 random으로 선정하여 current mean으로 설정 
    if(current_mean_random):
        current_mean= random.randint(0,len(D)-1)
        for i in range(0,5):
            current_mean_arr[0][i] = D[current_mean][i]
    below_threshold_arr=[]
    for i in range(0,len(D)):
        ecl_dist = 0
        color_total_current = 0
        color_total_new = 0
#random하게 선정된 row의 l2 distance 계산
        for j in range(0,5):
            ecl_dist += ((current_mean_arr[0][j] - D[i][j])**2)                
        ecl_dist = ecl_dist**0.5
#해당 distance가 threshold 내에 존재하는지 확인      
        if(ecl_dist < threshold):
            below_threshold_arr.append(i)    
    mean_R=0
    mean_G=0
    mean_B=0
    mean_i=0
    mean_j=0
    current_mean = 0
    mean_col = 0
    
#threshold 내에 있는 row들의 평균을 R,G,B 모두 나눠서 계산    
    for i in range(0, len(below_threshold_arr)):
        mean_R += D[below_threshold_arr[i]][0]
        mean_G += D[below_threshold_arr[i]][1]
        mean_B += D[below_threshold_arr[i]][2]
        mean_i += D[below_threshold_arr[i]][3]
        mean_j += D[below_threshold_arr[i]][4]   
    
    mean_R = mean_R / len(below_threshold_arr)
    mean_G = mean_G / len(below_threshold_arr)
    mean_B = mean_B / len(below_threshold_arr)
    mean_i = mean_i / len(below_threshold_arr)
    mean_j = mean_j / len(below_threshold_arr)
    
#average distance를 계산하여 현재의 mean과 비교
    mean_e_distance = ((mean_R - current_mean_arr[0][0])**2 + (mean_G - current_mean_arr[0][1])**2 + (mean_B - current_mean_arr[0][2])**2 + (mean_i - current_mean_arr[0][3])**2 + (mean_j - current_mean_arr[0][4])**2)    
    mean_e_distance = mean_e_distance**0.5        
    nearest_i = 0
    min_e_dist = 0
    counter_threshold = 0

    if(mean_e_distance < iter):                
        new_arr = np.zeros((1,3))
        new_arr[0][0] = mean_R
        new_arr[0][1] = mean_G
        new_arr[0][2] = mean_B
        for i in range(0, len(below_threshold_arr)):
            R[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = new_arr          
            D[below_threshold_arr[i]][0] = -1
        current_mean_random = True
        new_D=np.zeros((len(D),5))
        counter_i = 0
        
        for i in range(0, len(D)):
            if(D[i][0] != -1):
                new_D[counter_i][0] = D[i][0]
                new_D[counter_i][1] = D[i][1]
                new_D[counter_i][2] = D[i][2]
                new_D[counter_i][3] = D[i][3]
                new_D[counter_i][4] = D[i][4]
                counter_i += 1
                    
        D=np.zeros((counter_i,5))
        
        counter_i -= 1
        for i in range(0, counter_i):
            D[i][0] = new_D[i][0]
            D[i][1] = new_D[i][1]
            D[i][2] = new_D[i][2]
            D[i][3] = new_D[i][3]
            D[i][4] = new_D[i][4]
        
    else:
        current_mean_random = False
         
        current_mean_arr[0][0] = mean_R
        current_mean_arr[0][1] = mean_G
        current_mean_arr[0][2] = mean_B
        current_mean_arr[0][3] = mean_i
        current_mean_arr[0][4] = mean_j
        
    cv2.imwrite("image_filtered.png", R)
    

#(a)-1 mean shift filtering
image= cv2.imread("bread.jpg",1)
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imwrite("image_filtered.png", shifted)

#(a)-2 gaussian blur
pylab.rcParams['figure.figsize'] = 16, 12
image = Image.open('bread.jpg')

image = np.array(image)
original_shape = image.shape
X = np.reshape(image,(-1,3))

segmented_image = gaussian(image, sigma=5,multichannel=True)

plt.imshow(segmented_image)
plt.axis('off')
plt.savefig("image_"+"gaussian"+".png")

#(b) mean_shift_segmentation
pylab.rcParams['figure.figsize'] = 16, 12
image = Image.open('bread.jpg')

image = np.array(image)
original_shape = image.shape
X = np.reshape(image,(-1,3))

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

segmented_image = np.reshape(labels, original_shape[:2])

plt.imshow(segmented_image)
plt.axis('off')
plt.savefig("image_"+"segmented"+".png")
