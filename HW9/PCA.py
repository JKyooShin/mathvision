import numpy as np
import numpy.linalg as linalg
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans



path_dir = 'att_faces/'
file_list = os.listdir(path_dir)

lst = []
for png in file_list:
    print(png)
    if int(png.split('_')[1].split('.')[0]) == 1:
        pass
    else:
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
orpic = pic.reshape(360,2576)


def PCA(data, n):
    #정규화
    C = np.cov(data.T)
    #공분산
    evals, evecs  =linalg.eig(C)
    #고유값과 고유 벡터를 뽑아낸다.
    indices = np.argsort(evals)[::-1]

    evecs = evecs[:, indices]
    evals = evals[indices]
    # 큰 고유값 순서대로 정렬한다.

    if n > 0 :
        evecs =evecs[:,:n]
    #정렬된 고유벡터를 축소시킬 차원만큼만 indices 만큼만 유지한다.

    ratio = sum(evals[:n]/sum(evals))
    #주성분의 변화량이 전체 변화량을 설명하는 정도.

    x = np.dot(data, evecs)
    #표준화된 데이터에 고유벡터행렬을 곱해 고유공간으로 projection
    y = np.dot(x, evecs.T)
    #변환된 데이터에 역변환행렬을 곱해 원래의 데이터로 근사.
    return np.array([data, x, y ,evals, evecs, ratio])

data = PCA(orpic,10)
evecs=(data[4])

print(data[5])
print("복원되는 정도 : {}".format(data[5]))
photo= (data[4])
print(data[3].shape)
print(data[4].shape)
photo = np.array(photo, dtype=float)





photo= (data[4].T)

photo = photo.reshape(10,56,46)
photo = np.array(photo, dtype=float)
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(photo[i], cmap=plt.cm.bone)
plt.show()
plt.clf()







data = PCA(orpic,1)
evecs=(data[4])
print("복원되는 정도 : {}".format(data[5]))

lst = []
for png in file_list:
    if png.split('_')[0] == 's1':
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
pic = pic.reshape(10,2576)
x=np.dot(pic, evecs)
print(x.shape)
y=np.dot(x, evecs.T)
y = y.reshape(10, 56,46)
y = np.array(y, dtype=float)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(y[i], cmap=plt.cm.bone)
plt.show()
plt.clf()

data = PCA(orpic,10)
evecs=(data[4])
print("복원되는 정도 : {}".format(data[5]))

lst = []
for png in file_list:
    if png.split('_')[0] == 's1':
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
pic = pic.reshape(10,2576)
x=np.dot(pic, evecs)
print(x.shape)
y=np.dot(x, evecs.T)
y = y.reshape(10, 56,46)
y = np.array(y, dtype=float)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(y[i], cmap=plt.cm.bone)
plt.show()
plt.clf()




data = PCA(orpic,100)
evecs=(data[4])
print("복원되는 정도 : {}".format(data[5]))

lst = []
for png in file_list:
    if png.split('_')[0] == 's1':
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
pic = pic.reshape(10,2576)
x=np.dot(pic, evecs)
print(x.shape)
y=np.dot(x, evecs.T)
y = y.reshape(10, 56,46)
y = np.array(y, dtype=float)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(y[i], cmap=plt.cm.bone)
plt.show()
plt.clf()



data = PCA(orpic,200)
evecs=(data[4])
print("복원되는 정도 : {}".format(data[5]))

lst = []
for png in file_list:
    if png.split('_')[0] == 's1':
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
pic = pic.reshape(10,2576)
x=np.dot(pic, evecs)
print(x.shape)
y=np.dot(x, evecs.T)
y = y.reshape(10, 56,46)
y = np.array(y, dtype=float)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(y[i], cmap=plt.cm.bone)
plt.show()
plt.clf()

data = PCA(orpic,1000)

evecs=(data[4])
print("복원되는 정도 : {}".format(data[5]))

lst = []
for png in file_list:
    if png.split('_')[0] == 's1':
        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)
pic = np.stack(lst)
pic = pic.reshape(10,2576)
x=np.dot(pic, evecs)
print(x.shape)
y=np.dot(x, evecs.T)
y = y.reshape(10, 56,46)
y = np.array(y, dtype=float)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(y[i], cmap=plt.cm.bone)
plt.show()
plt.clf()
