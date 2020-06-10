import numpy as np
import matplotlib.pyplot as plt


def f(x,y):
    return np.sin(x+y-1) + (x-y-1)**2- 1.5*x+2.5*y+1

X = (np.linspace(1.0 , 5.0, num=1000))
Y = (np.linspace(-3.0 , 4.0, num=1000))

from mpl_toolkits.mplot3d import Axes3D


a, b = np.meshgrid(X,Y)
c = f(a,b)
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(a, b, c, alpha=0.5)
#plt.show()

import sympy
x, y = sympy.symbols('x y')

dif_x  = sympy.diff((sympy.sin(x+y-1) + (x-y-1)*(x-y-1)- 1.5*x+2.5*y+1), x)
dif_y = sympy.diff((sympy.sin(x+y-1) + (x-y-1)*(x-y-1)- 1.5*x+2.5*y+1), y)

xs_q, ys_q = np.meshgrid(np.linspace(1.0, 5.0,num=10), np.linspace(-3.0, 4.0,num=10))



#xs_q_grad = [float(dif_x.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]
#ys_q_grad = [float(dif_y.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]

#print(len(xs_q_grad))
init_x= float(np.random.randn(1))+np.random.randint(2,4)
init_y= float(np.random.randn(1))+np.random.randint(-2,3)
xk = np.array([init_x, init_y])
lam = 0.1
number = 0

while True:
    dist = lam*np.array([float(dif_x.subs(x, xk[0]).subs(y, xk[1])),float(dif_y.subs(x, xk[0]).subs(y, xk[1]))])
    xk= xk - dist
    number+=1
    ax.scatter(xk[0], xk[1], f(xk[0], xk[1]), marker='o', s=10)
    if np.sqrt(dist[0]**2+dist[1]**2)< 0.00001:
        print(xk)
        print(number)
        print(f(xk[0], xk[1]))
        break


number = 0
H = np.array([[sympy.diff(dif_x, x), sympy.diff(dif_x, y), ],[sympy.diff(dif_y, x) , sympy.diff(dif_y, y)]])
xk = np.array([init_x, init_y])
while True:
    hk = np.array([[float(H[0, 0].subs(x, xk[0]).subs(y, xk[1])),
                    float(H[0, 1].subs(x, xk[0]).subs(y, xk[1]))],
                   [float(H[1, 0].subs(x, xk[0]).subs(y, xk[1])),
                    float(H[1, 1].subs(x, xk[0]).subs(y, xk[1]))]])
    invhk = np.linalg.pinv(hk)
    dist = invhk @ np.array([float(dif_x.subs(x, xk[0]).subs(y, xk[1])), float(dif_y.subs(x, xk[0]).subs(y, xk[1]))])
    xk = xk - dist
    number +=1
    ax.scatter(xk[0], xk[1], f(xk[0], xk[1]), marker='x', s=10)
    if np.sqrt(dist[0]**2+dist[1]**2)< 0.00001:
        print(xk)
        print(number)
        print(f(xk[0], xk[1]))
        break


plt.show()


