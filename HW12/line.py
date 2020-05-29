import numpy as np
import cv2
from numpy.linalg import svd
import matplotlib
import matplotlib.pyplot as plt
from ransac import *

events = [i for i in dir(cv2) if 'EVENT' in i]
click = False
x1, y1 = -1, -1

def augment(xys):

    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy


def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click, ptlst,dot_num,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])
        
        
        
        numdot = 20 # 누를 점 숫자.
        
        
        
        if len(ptlst) == numdot:

            pts= np.array(ptlst)

            n = numdot
            max_iterations = 100
            goal_inliers = n * 0.3

            for i in range(numdot):
                cv2.line(img, (int(pts[ i, 0 ]), int(pts[i, 1 ])),(int(pts[ i, 0 ]), int(pts[i , 1])), (255, 255, 255), 5)

            m, b, s = run_ransac(pts, estimate, lambda x, y: is_inlier(x, y, 0.1), 2, goal_inliers, 5)
            a, b, c = m
            for dot in s:
                cv2.line(img, (int(dot.T[0]), int(dot.T[1])), (int(dot.T[0]), int(dot.T[1])),(255, 0, 0), 5)
            cv2.line(img, (0, int(-c / b)) ,(1000, int( -(c + 1000 * a) / b)) , (255, 0, 0), 2)

            m, b, s = run_ransac(pts, estimate, lambda x, y: is_inlier(x, y, 0.05), 2, goal_inliers, 5)
            a, b, c = m
            for dot in s:
                cv2.line(img, (int(dot.T[0]), int(dot.T[1])), (int(dot.T[0]), int(dot.T[1])),(255, 0, 255), 5)
            cv2.line(img, (0, int(-c / b)) ,(1000, int( -(c + 1000 * a) / b)) , (255, 0, 255), 2)

            '''

            for i in range(numdot):
                cv2.line(img, (int(pts[ i, 0 ]), int(pts[i, 1 ])),(int(pts[ i, 0 ]), int(pts[i , 1])), (255, 255, 255), 5)
            for iter in range(10):
                m, b, s = run_ransac(pts, estimate, lambda x, y: is_inlier(x, y, 0.1), 2, goal_inliers, 10+iter*5)
                a, b, c = m
                for dot in s:
                    cv2.line(img, (int(dot.T[0]), int(dot.T[1])), (int(dot.T[0]), int(dot.T[1])), (255, 0, iter*25), 5)
                cv2.line(img, (0, int(-c / b)), (1000, int(-(c + 1000 * a) / b)), (0,0,iter*25), 2)

            ptlst=[]
            end = True
            '''
            #cv2.putText(img, str(abs(volume)), (100,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,255,0), 2 )

def residual(X, Y, a, b, n):
    res =[]
    for i in range(n):
        res.append(Y[i] - (a*X[i]+b))
    R = np.stack(res)
    w=[]
    for i in range(n):
        w.append(1/ ((abs(R[i])/1.3998) +1))
    W = np.stack(w)
    return np.diag(W)



img = np.zeros((1000, 1000, 3), np.uint8)
ptlst =[]
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

end =False
print("Numdots 의 숫자를 수정해줘서 누를 점을 바꿔주세요. default : 20")
while True:
    cv2.imshow('image', img)  # 화면을 보여준다.
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
