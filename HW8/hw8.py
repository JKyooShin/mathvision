import numpy as np
from sklearn.decomposition import PCA
import matplotlib .pyplot as plt
import scipy.stats as st


def openfile(path):
    a = open(path)
    b = a.readlines()
    x = []
    for i in b:
        y = []
        c = (i.replace('\n', '').split(','))
        for j in c:
            y.append(float(j))
        x.append(y)
    x = np.asarray(x)
    return x
dta = openfile('data_a.txt')
dtb = openfile('data_b.txt')
dtst = openfile('test.txt')

dt = (np.concatenate((dta, dtb, dtst), axis=0))
pca = PCA(n_components=2)
D = pca.fit_transform(dt)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

test = (D[1500:1502,:])
plt.scatter(D[0:1000, 0], D[0:1000, 1])
plt.scatter(D[1000:1500, 0], D[1000:1500, 1])

plt.show()

# Extract x and y
xa = D[0:1000, 0]
ya = D[0:1000, 1]
# Define the borders
deltaX = (max(xa) - min(xa))/10
deltaY = (max(ya) - min(ya))/10
xmin = min(xa) - deltaX
xmax = max(xa) + deltaX
ymin = min(ya) - deltaY
ymax = max(ya) + deltaY
xxa, yya = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xxa.ravel(), yya.ravel()])
values = np.vstack([xa, ya])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xxa.shape)


x = D[1000:1500, 0]
y = D[1000:1500, 1]
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f2 = np.reshape(kernel(positions).T, xx.shape)
cfset = plt.contour(xxa, yya, f, cmap='Spectral')
cfset = plt.contour(xx, yy, f2, cmap='coolwarm')
plt.show()
mean = D[0:1000, :].mean(0)
cov = np.cov(D[0:1000:,:].T)
mean2 = D[1000:1500, :].mean(0)
cov2 = np.cov(D[1000:1500:,:].T)

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 5000).T
plt.plot(x, y, 'x')
plt.plot(x2, y2, 'x')
plt.axis('equal')
plt.show()

print(((test[0,:]-mean)) @np.linalg.inv(cov)@(test[0,:]-mean).T)

print(((test[0,:]-mean2)) @np.linalg.inv(cov2)@(test[0,:]-mean2).T)

print(((test[1,:]-mean)) @np.linalg.inv(cov)@(test[1,:]-mean).T)

print(((test[1,:]-mean2)) @np.linalg.inv(cov2)@(test[1,:]-mean2).T)
