import cv2
import numpy as np
import time


def get_circle(dt):
    dt = dt[0,:,:]
    dot_num= dt.shape[0]
    data = np.zeros([3,dot_num])
    asnw = np.zeros([1,dot_num])
    for i in range(dot_num):
        x = dt[i, 0]
        y = dt[i, 1]
        data[0, i] = -2*x
        data[1, i] = -2*y
        data[2, i] = 1

        asnw[0, i]= -(x**2+y**2)
    a = (np.linalg.pinv(data))

    return (asnw@a)[0]







def draw(event, x, y, flags, param):
    global x1, y1, click, ptlst,dot_num,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])


        print("점 설정 : (" + str(x1) + ", " + str(y1) + ")")

        if len(ptlst) ==int(dot_num):

            center = (get_circle(np.int32([ptlst])))
            a = center[0]
            b= center[1]
            r = np.sqrt(a**2+b**2-center[2])

            cv2.circle(img, (int(a),int(b)), int(r)  , (255, 0, 0) ,2, -1)
            cv2.polylines(img, np.int32([ptlst]), True, (0,255,0), 2)
            ptlst=[]
            end = True




dot_num =input("write in number of dots")


img = np.zeros((500, 500, 3), np.uint8)
ptlst =[]
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

end =False

while True:
    cv2.imshow('image', img)  # 화면을 보여준다.
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
