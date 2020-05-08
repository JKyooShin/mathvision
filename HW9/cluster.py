import numpy as np
import numpy.linalg as linalg
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

path_dir = 'att_faces/'
file_list = os.listdir(path_dir)

lst = []
first=[]
name1 = []
name2= []
for png in file_list:
    if int(png.split('_')[1].split('.')[0]) == 1:
        name1.append(png.split('_')[0])

        image = Image.open(path_dir + png)
        pixel = np.array(image)
        first.append(pixel)
    else:
        name2.append(png.split('_')[0])

        image = Image.open(path_dir + png)
        pixel = np.array(image)
        lst.append(pixel)


pic = np.stack(lst)
orpic = pic.reshape(360,2576)
orpic = np.array(orpic, dtype=float)

fpic = np.stack(first)
frpic =fpic.reshape(40,2576)
frpic = np.array(frpic, dtype=float)

def PCA(data, n):
    from sklearn.preprocessing import StandardScaler
    data = StandardScaler().fit_transform(data)
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



data = PCA(orpic,50)
evecs=(data[4])
orpic = np.dot(orpic, evecs)
print(orpic.shape)
frpic = np.dot(frpic, evecs)
def recog():
    cnt=[]
    for i in range(40):
        res = m(i)
        cnt.append(res)
    return cnt
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
def m(test):
    global orpic, frpic
    rorpic = orpic.real
    rfrpic = frpic.real
    distance =[]
    for i in range(360):
        distance.append(dist(rfrpic[test, :], rorpic[i, :]))
    return distance.index(min(distance))
def mypic(test):
    global orpic, frpic
    rorpic = orpic.real
    distance =[]
    for i in range(360):
        distance.append(dist(test, rorpic[i, :]))
    return distance.index(min(distance))
ress = (recog())
print(ress)
nct = 0
for i,  r in enumerate(ress):
    print(name2[r])
    print(name1[i])
    if name2[r] == name1[i]:
        nct+=1
print(nct/40)


image = Image.open('neo293.jpg').convert('L')
pixel = np.array(image)
pixel = np.array(pixel, dtype=float)
pixel = pixel.flatten().reshape(1,-1)
x=np.dot(pixel, evecs)

rr = mypic(x)
print(rr)
print(name2[rr])


