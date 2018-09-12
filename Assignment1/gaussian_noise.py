import numpy as np
import math
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io

def plot(samples, img_num, channels):
    fig = plt.figure(figsize = (3*img_num,5))
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

def add_gaussian_noise(image, sigma):
    pi1 = np.random.uniform(low=0.0,high=1.0,size=(512,513))
    r1 = np.random.uniform(low=0.0,high=1.0,size=(512,513))


    cos1 = np.cos(np.multiply(2*math.pi,pi1))
    root1 = np.sqrt(-2 * np.log(r1))
    z1 = np.multiply(sigma,np.multiply(cos1,root1))

    cos2 = np.sin(np.multiply(2*math.pi,pi1))
    root2 = np.sqrt(-2 * np.log(r1))
    z2 = np.multiply(sigma,np.multiply(cos2,root2))

    npad1 = ((0,1),(0,0))
    npad2 = ((1,0),(0,0))
    z1_pad = np.pad(z1,npad1,'constant', constant_values=(0))
    z2_pad = np.pad(z2,npad2,'constant', constant_values=(0))

    noise = np.add(z1_pad,z2_pad)[1:513,0:512]
    img_noised = np.add(image,noise)
    img_noised = np.clip(img_noised,0.0,255)

    return img_noised

def estimate_sigma(cropped):
    variance = np.var(cropped)
    return math.sqrt(variance)
    


sigma_set =[5.0,10.0,15.0, 20.0, 25.0, 30.0 ]
img1=io.imread('lena_gray.bmp')
#img3 = io.imread('barbara_gray.bmp')
img2=io.imread('camera_gray.bmp')

plt.show(block = False)
img_set = []
img1_set=[]
img2_set= []
img_set.append(img1)
for sigma in sigma_set:
    img_noised = add_gaussian_noise(img1,sigma)
    img_set.append(img_noised)
    img1_set.append(img_noised)
img_set.append(img2)
for sigma in sigma_set:
    img_noised = add_gaussian_noise(img2,sigma)
    img_set.append(img_noised)
    img2_set.append(img_noised)
#img_set.append(img3)
#for sigma in sigma_set:
#    img_noised = add_gaussian_noise(img3,sigma)
#    img_set.append(img_noised)
fig = plot(img_set,len(sigma_set)+1,1)    
plt.show()

print('Problem 1')
print()
print('[ image 1 ]')
for i, sigma in enumerate(sigma_set):
    estimated = estimate_sigma(np.subtract(img1_set[i],img1))
    print('Original: {}, Estimated: {}'.format(sigma, estimated))
print()    
print('[ image 2 ]')
for i, sigma in enumerate(sigma_set):
    estimated = estimate_sigma(np.subtract(img2_set[i],img2))
    print('Original: {}, Estimated: {}'.format(sigma, estimated))
print()    
print('Problem 2')
print()
print('[ image 1 ]')
print()
best1 = [2000,0,0,0,0]
for sigma in sigma_set:
    best1.append(sigma)
for it in range(5):
    for jt in range(5):
        print('[ {}:{} , {}:{} ]'.format(100*it,100*(it+1),100*jt,100*(jt+1)))
        diff = 0.0
        temp = []
        for i, sigma in enumerate(sigma_set):
            estimated = estimate_sigma(img1_set[i][100*it:100*(it+1),100*jt:100*(jt+1)])
            print('Original: {}, Estimated: {}'.format(sigma, estimated))
            temp.append(estimated)
            diff += estimated - sigma
        if diff < best1[0]:
            best1 = [diff,100*it,100*(it+1),100*jt,100*(jt+1)]
            for est in temp:
                best1.append(est)
        print()
print()
print('[ image 2 ]')
print()
best2 = [2000,0,0,0,0]
for sigma in sigma_set:
    best2.append(sigma)
for it in range(5):
    for jt in range(5):
        print('[ {}:{} , {}:{} ]'.format(100*it,100*(it+1),100*jt,100*(jt+1)))
        diff = 0.0
        temp = []
        for i, sigma in enumerate(sigma_set):
            estimated = estimate_sigma(img2_set[i][100*it:100*(it+1),100*jt:100*(jt+1)])
            print('Original: {}, Estimated: {}'.format(sigma, estimated))
            temp.append(estimated)
            diff += estimated - sigma
        if diff < best2[0]:
            best2 = [diff,100*it,100*(it+1),100*jt,100*(jt+1)]
            for est in temp:
                best2.append(est)
        print()

print("best patch for image 1: [ {}:{} , {}:{} ]".format(best1[1],best1[2],best1[3],best1[4]))
for i, sigma in enumerate(sigma_set):
    estimated = best1[5+i]
    print('Original: {}, Estimated: {}'.format(sigma, estimated))
print()
print("best patch for image 2: [ {}:{} , {}:{} ]".format(best2[1],best2[2],best2[3],best2[4]))
for i, sigma in enumerate(sigma_set):
    estimated = best2[5+i]
    print('Original: {}, Estimated: {}'.format(sigma, estimated))
