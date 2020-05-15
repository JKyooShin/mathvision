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
        if len(ptlst) == int(dot_num):
            n=int(dot_num)
            pts= np.array(ptlst).T
            A = np.asarray([pts[0, :], np.ones(n)])

            U, S, VT = svd(A)
            s = np.array([0])
            e = np.array([1000])
            iplace = 0
            for i in range(2):
                if S[i] == 0:
                    iplace = i
            if iplace != 0:
                inverse = VT.T[:, iplace]
                Start = np.array([s[0], inverse[0] * s[0] + inverse[1]])
                End = np.array([e[0], inverse[0] * e[0] + inverse[1]])
            else:
                inverse = (VT.T @ (np.diag(1 / S, n - 2)[:, n - 2:]) @ U.T)
                inverse = (pts[1, :] @ inverse)
                Start = [s[0], inverse[0] * s[0] + inverse[1]]
                End = [e[0], inverse[0] * e[0] + inverse[1]]
            #cv2.polylines(img, np.int32([ptlst]),True, (255,0,0), 2)

            B = np.asarray([pts[0, :], pts[1, :], np.ones(n)])
            U, S, VT = svd(B.T)
            s = np.array([0])
            e = np.array([1000])
            inverse = VT.T[:,-1]
            Start2 = np.array([s[0], (inverse[0]*s[0]+inverse[2])*(-1/inverse[1])])
            End2 = np.array([e[0], (inverse[0]*e[0]+inverse[2])*(-1/inverse[1])])
            print(pts.shape)
            print(n)
            for i in range(n):
                cv2.line(img, (int(pts[0, i]), int(pts[1,i])),(int(pts[0, i]), int(pts[1,i])), (255, 255, 255), 5)

            cv2.line(img ,(int(Start[0]), int(Start[1])) , (int(End[0]),int(End[1])) , (0,255,0))
            cv2.line(img ,(int(Start2[0]), int(Start2[1])) , (int(End2[0]),int(End2[1])) , (0,0,255))

            ptlst=[]
            end = True
            #cv2.putText(img, str(abs(volume)), (100,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,255,0), 2 )




dot_num =input("write in number of dots")


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
