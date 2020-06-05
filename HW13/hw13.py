#𝑧 = 𝑓 𝑥, 𝑦 = (𝑥 + 𝑦)(𝑥𝑦 + 𝑥𝑦)


import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x+y)*(x*y+x*y*y)

xn=10
a = np.arange(-1, 1.6, 0.1)
A=a
b = np.arange(-1.2, 0.3, 0.1)
B=b
from mpl_toolkits.mplot3d import Axes3D


a, b = np.meshgrid(a, b)
c = f(a,b)
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(a, b, c, alpha=0.5)
ax.scatter(1, 0)
plt.show()



print(a.shape)
def fx(x, y):
    return x*y**2 + x*y + (x + y)*(y**2 + y)
def fy(x, y):
    return x*y**2 + x*y + (x + y)*(2*x*y + x)

print(f(1,0))
print(fx(1,0))
print(fy(1,0))

plt.plot(A, f(A,0))
plt.scatter(1, f(1,0))
plt.show()
plt.plot(B,f(1,B))
plt.scatter(0, f(1,0))
plt.show()

def f_x_x(x,y):
    return 2*y+2*y**2
def f_x_y(x,y):
    return x+2*x*y+y+y**2+x+y+2*x*y+2*y*y
def f_y_x(x,y):
    return x+ 2*x*y + y + y*y +x+y+2*x*y+2*y*y
def f_y_y(x,y):
    return 2*x+4*x*y+2*x*(x+y)


def h(x,y):
    return np.array([[f_x_x(x,y), f_x_y(x,y)],[f_y_x(x,y), f_y_y(x,y)]])

print(np.linalg.eigvals(h(0,0)))
print(np.linalg.eigvals(h(0,-1)))
print(np.linalg.eigvals(h(1,-1)))
print(np.linalg.eigvals(h(3/8,-3/4)))


fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(a, b, c, alpha=0.5)
ax.scatter(0, 0)
ax.scatter(0, -1)
ax.scatter(1, -1)
ax.scatter(3/8, -3/4)
plt.show()
