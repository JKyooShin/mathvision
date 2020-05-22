import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('hw11_sample.png',cv.IMREAD_GRAYSCALE)

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

im = np.asarray(img)
print(im.shape[0])
f=[]
y=[]
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        f.append([
            np.array(i ** 2),
            np.array(j ** 2),
            np.array(i * j),
            np.array(i),
            np.array(j),
            np.array(1)
        ])
        y.append(im[i,j])
f =np.stack(f)
Y = np.array(y)

F = np.linalg.pinv(f)
[a,b,c,d,e,f] = (F@Y)

back = np.zeros([im.shape[0],im.shape[1]])
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        back[i,j] =a*i**2 + b*j**2 + c*i*j + d*i + e*j + f
print(im.shape)
print(back.shape)
plt.imshow(back, cmap=plt.get_cmap('gray'))
plt.show()

plt.imshow(im-back, cmap=plt.get_cmap('gray'))
plt.show()
