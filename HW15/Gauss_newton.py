import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sympy

np.random.seed(4)
a, b, c, d = sympy.symbols('a, b, c, d')
x, y = sympy.symbols('x, y')

f = a*sympy.sin(b*x + c) + d - y

A = sympy.diff(f, a)
B = sympy.diff(f, b)
C = sympy.diff(f, c)
D = sympy.diff(f, d)

N=10
sample_x = np.linspace(-20, 20, 100)
#target = np.random.randint(0, 99, 75)
#addr=np.array([ 5,6,7,8,9,17,26, 27,28,29,46,47,48,49,66,67,68,69,86,87,88,89])
#sample_x = sample_x[np.concatenate([target, addr])]
randA = np.abs(np.random.randn(1))
randB = np.abs(np.random.randn(1))
randC = np.abs(np.random.randn(1))
randD = np.abs(np.random.randn(1))
print(randA, randB, randC, randD)
sample_y = randA*np.sin(sample_x*randB+randC)+randD + np.random.randn(1)[0]
smaple_dots=np.stack([sample_x, sample_y]).T




F = []
Ja=[]

for i in range(N):
    x_sample = smaple_dots[i, 0]
    y_sample = smaple_dots[i, 1]

    F.append(f.subs(x, x_sample).subs(y, y_sample))
    Ja.append([A.subs(x, x_sample).subs(y, y_sample), B.subs(x, x_sample).subs(y, y_sample), C.subs(x, x_sample).subs(y, y_sample), D.subs(x, x_sample).subs(y, y_sample)])

Ja = np.stack(Ja)
#자코비안 행렬
F=np.stack(F)
#F 행렬



#초기값 설정

def F_update(F, X):
    res = []
    for i in F:
        res.append(i.subs(a, X[0]).subs(b, X[1]).subs(c, X[2]).subs(d, X[3]))
    res= np.stack(res)
    return res

def Ja_update(Ja, X):
    res = []
    for i in range(N):
        tmp=[]
        for j in range(len(X)):
            tmp.append(Ja[i, j].subs(a, X[0]).subs(b, X[1]).subs(c, X[2]).subs(d, X[3]))
        res.append(tmp)
    res = np.stack(res)
    return res
def fnc(X_k, x):
    return X_k[0]*np.sin(X_k[1]*x * X_k[2]) + X_k[3]

#X_k = [np.random.randn(1),np.random.randn(1),np.random.randn(1),np.random.randn(1)]
X_k = [np.std(smaple_dots[:,1]),0.01,1,np.mean(smaple_dots[:,1])]

update = 1
cnt = 0
while  True:
    J_k = Ja_update(Ja, X_k)
    F_k = F_update(F, X_k)
    tmp = (J_k.T @J_k)
    tmp = np.array(tmp, dtype='float')
    Jinv = np.linalg.pinv(tmp)
    update = Jinv@J_k.T@F_k
    X_k = X_k - 0.1*update
    cnt +=1
    #if (np.sum(np.abs(update))) < 1E-9:

    if cnt%100 == 0:
        print(cnt)
        X_k = np.array(X_k, dtype='float')
        print(X_k)
        x = np.linspace(-20, 20, 1000)
        y = fnc(X_k, x)
        plt.scatter(smaple_dots[:, 0], smaple_dots[:, 1], c='red')
        plt.plot(x, y)
        plt.show()
