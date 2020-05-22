import numpy as np
import cv2
from numpy.linalg import svd


events = [i for i in dir(cv2) if 'EVENT' in i]
click = False
x1, y1 = -1, -1


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click, ptlst,dot_num,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])
        if len(ptlst) == 5:
            n=5
            pts= np.array(ptlst)
            print(pts.shape)

            f = []
            y = []
            for i in range(5):
                f.append([
                    np.array(pts[i, 0]),
                    np.array(1)
                ])
                y.append([
                    np.array(pts[i, 1])
                ])
            A=np.stack(f)
            Y = np.stack(y)
            F = np.linalg.pinv(A)

            [a, b] = (F @ Y)
            Start = [0, a[0] * 0 + b[0]]
            End = [1000, a[0] * 1000 + b[0]]
            a= a[0]
            b=b[0]
            cv2.line(img, (int(Start[0]), int(Start[1])), (int(End[0]), int(End[1])), (0, 255, 0))
            for i in range(n):
                cv2.line(img, (int(pts[ i, 0 ]), int(pts[i, 1 ])),(int(pts[ i, 0 ]), int(pts[i , 1])), (255, 255, 255), 5)
            for iter in range(15):
                print(iter)
                W = residual(pts[:, 0], pts[:, 1], a, b, n)
                X1 = np.linalg.pinv((A.T @ W @ A)) @ (A.T @ W @ Y)
                a = X1[0, 0]
                b = X1[1, 0]
                Start = [0, a * 0 + b]
                End = [1000, a * 1000 + b]
                if iter%3 == 0:
                    cv2.line(img, (int(Start[0]), int(Start[1])), (int(End[0]), int(End[1])), (255, 0, 0))
            ptlst=[]
            end = True
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

while True:
    cv2.imshow('image', img)  # 화면을 보여준다.
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
