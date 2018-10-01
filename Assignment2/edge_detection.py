import numpy as np
import math as m
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io, measure
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

def plot(samples, img_num, channels):
    fig = plt.figure(figsize = (3*img_num,4))
    gs = gridspec.GridSpec(2,img_num)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channels == 1:
            plt.imshow(sample, cmap='Greys_r')
        else:
            plt.imshow(sample)         
    return fig

def edge_detector(image):
    dx = ndimage.sobel(image,axis=0)
    dy = ndimage.sobel(image,axis=1)
    result = np.abs(dx) + np.abs(dy)

    return result

def edge_suppressor(image):
    dx = ndimage.sobel(image,axis=0)
    dy = ndimage.sobel(image,axis=1)
    magnitude = np.abs(dx) + np.abs(dy)
    marker = np.zeros((len(dx),len(dx)))        
    direction = np.arctan(np.divide(dy,dx))
    for i in range(len(dx)):
        for j in range(len(dx)):
            curr_dirc = direction[i][j]
            if curr_dirc > -m.pi*0.25 and curr_dirc < m.pi*0.25:
                pix1 = (i,j+1)
                pix2 = (i,j-1)                
            elif curr_dirc <= -m.pi*0.25  and curr_dirc >= -m.pi*0.75:
                pix1 = (i+1,j+1)
                pix2 = (i-1,j-1)

            elif curr_dirc >= m.pi*0.25 and curr_dirc <= m.pi*0.75:
                pix1 = (i+1,j-1)
                pix2 = (i-1,j+1)
            else:
                pix1 = (i+1,j)    
                pix2 = (i-1,j)
            if (pix1[0] >= 0) and (pix1[1] >= 0) :
                if (pix1[0] < len(dx)) and (pix1[1] < len(dx)) :
                    if magnitude[pix1[0],pix1[1]] > magnitude[i,j]:
                        marker[i,j]=1
            if (pix2[0] >= 0) and (pix2[1] >= 0):
                if (pix2[0] < len(dx)) and (pix2[1] < len(dx)):
                    if magnitude[pix2[0],pix2[1]] > magnitude[i,j]:
                        marker[i,j]=1                    
    for i in range(len(dx)):
        for j in range(len(dx)):
            if marker[i][j] == 1:
                magnitude[i][j] = 0
            #else:
                #magnitude[i][j] = 255
    #print(marker)
    return magnitude

def hysteresis(image, low, high):
     mag = edge_suppressor(image)
     marker = np.zeros((len(mag),len(mag)))
     for i in range(len(mag)):
         for j in range(len(mag)):
             if mag[i][j] > high:
                 marker[i][j]= 1
     for i in range(len(mag)):
         for j in range(len(mag)):
             if mag[i][j] > low:
                 try:
                     if marker[i-1][j] == 1 and marker[i+1][j] == 1:
                         marker[i][j]= 1
                     elif marker[i][j-1] == 1 and marker[i][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i-1][j-1] == 1 and marker[i+1][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i+1][j-1] == 1 and marker[i-1][j+1] ==1:
                         marker[i][j]= 1                         
                 except:
                     pass
     for i in range(len(mag)):
         for j in range(len(mag)):
             if mag[i][j] > low:
                 try:
                     if marker[i-1][j] == 1 and marker[i+1][j] == 1:
                         marker[i][j]= 1
                     elif marker[i][j-1] == 1 and marker[i][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i-1][j-1] == 1 and marker[i+1][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i+1][j-1] == 1 and marker[i-1][j+1] ==1:
                         marker[i][j]= 1                         
                 except:
                     pass
     for i in range(len(mag)):
         for j in range(len(mag)):
             if mag[i][j] > low:
                 try:
                     if marker[i-1][j] == 1 and marker[i+1][j] == 1:
                         marker[i][j]= 1
                     elif marker[i][j-1] == 1 and marker[i][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i-1][j-1] == 1 and marker[i+1][j+1] ==1:
                         marker[i][j]= 1
                     elif marker[i+1][j-1] == 1 and marker[i-1][j+1] ==1:
                         marker[i][j]= 1                         
                 except:
                     pass                    
     for i in range(len(mag)):
         for j in range(len(mag)):
             if marker[i][j] == 1:
                 mag[i][j] = 0
             else:
                 mag[i][j] = 255
     return mag

    
def border_detection(image, low, high):
    mag = hysteresis(image, low, high)
    dx = ndimage.sobel(image,axis=0)
    dy = ndimage.sobel(image,axis=1)      
    direction = np.arctan(np.divide(dy,dx))    
    path_dict = {}
    optimal_list = []
    for idx in range(len(mag)//10):
        for jdx in range(len(mag)//10):
            for i in range(50):
                for j in range(50):
                    max_mag = 0
                    max_cord = (0,0)
                    if i+idx*10 <512 and j+jdx*10 <512:
                        if mag[i+idx*10][j+jdx*10] > max_mag:
                            max_mag = mag[i+idx*10][j+jdx*10]
                            max_cord = (i+idx*10, j+jdx*10)
            if max_mag != 0:
                path_dict[str(max_cord[0])+','+str(max_cord[1])] = [max_cord]
    for start,path in path_dict.items():
        ix = int(start.split(',')[0])
        jx = int(start.split(',')[1])
        cost = mag[ix][jx]
        for it in range(10):
            dirc = direction[ix][jx]
            pix = []
            if dirc > -m.pi*0.25 and dirc < m.pi*0.25:
                pix.append((ix+1,jx+1))
                pix.append((ix+1,jx))                
                pix.append((ix+1,jx-1))
            elif dirc <= -m.pi*0.25  and dirc >= -m.pi*0.75:
                pix.append((ix+1,jx))
                pix.append((ix+1,jx-1))                
                pix.append((ix,jx-1))
            elif dirc >= m.pi*0.25 and dirc <= m.pi*0.75:
                pix.append((ix+1,jx))
                pix.append((ix+1,jx+1))                
                pix.append((ix,jx+1))              
            else:
                pix.append((ix+1,jx-1))   
                pix.append((ix,jx-1))                            
                pix.append((ix-1,jx-1))

            partial_max= 0
            partial_cord = (0,0)
            for pixel in pix:
                if pixel[0] < 0 or pixel[1] < 0 or pixel[0]>=512 or pixel[1]>=512 or mag[pixel[0]][pixel[1]]==0:
                    break
                else:
                    if mag[pixel[0]][pixel[1]] > partial_max:
                        partial_max = mag[pixel[0]][pixel[1]]
                        partial_cord = pixel
            if partial_max == 0 or (ix == pixel[0] and jx == pixel[1]):
                #final_path = path.append(cost)
                break
            else:
                path.append(partial_cord)
                ix = partial_cord[0]
                jx = partial_cord[1]
                #cost += partial_max
        optimal_list.append(path)                

    marker = np.zeros((len(mag),len(mag)))
    #optimal_list.sort(key = lambda path: len(path), reverse=True)
    #if len(optimal_list) > 10:
        #optimal_list= optimal_list[:10]
    for path in optimal_list:
        for cord in path:
            marker[cord[0]][cord[1]] = 255
            
    contours = measure.find_contours(mag, 0.8)            
    return contours 
                
        
img1=io.imread('lena_gray.bmp')
img2=io.imread('camera_gray.bmp')
img3 = io.imread('barbara_gray.bmp')

img1 = gaussian_filter(img1, sigma = 3)
img2 = gaussian_filter(img2, sigma = 3)
img3 = gaussian_filter(img3, sigma = 3)

plt.imshow(img1, cmap='Greys_r')
plt.show()
plt.imshow(edge_detector(img1), cmap='Greys_r')
plt.show()
plt.imshow(edge_suppressor(img1), cmap='Greys_r')
plt.show()
plt.imshow(hysteresis(img1,30,90), cmap='Greys_r')
plt.show()
contours = border_detection(img1,30,90)
fig, ax = plt.subplots()
ax.imshow(img1, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    if len(contour) > 30:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
plt.imshow(img2, cmap='Greys_r')
plt.show()
plt.imshow(edge_detector(img2), cmap='Greys_r')
plt.show()
plt.imshow(edge_suppressor(img2), cmap='Greys_r')
plt.show()
plt.imshow(hysteresis(img2,10,70), cmap='Greys_r')
plt.show()
contours = border_detection(img2,30,90)
fig, ax = plt.subplots()
ax.imshow(img2, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    if len(contour) > 30:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
plt.imshow(img3, cmap='Greys_r')
plt.show()
plt.imshow(edge_detector(img3), cmap='Greys_r')
plt.show()
plt.imshow(edge_suppressor(img3), cmap='Greys_r')
plt.show()
plt.imshow(hysteresis(img3,10,70), cmap='Greys_r')
plt.show()
contours = border_detection(img3,30,90)
fig, ax = plt.subplots()
ax.imshow(img3, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    if len(contour) > 30:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#fig = plot(img_set,len(sigma_set)+1,1)    
#plt.show()


