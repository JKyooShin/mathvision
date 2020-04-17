import cv2
import numpy as np
import time


def get_circle(dt):
    dt = dt[0,:,:]
    dot_num= dt.shape[0]
    data = np.zeros([5,dot_num])
    asnw = np.zeros([1,dot_num])
    for i in range(dot_num):
        x = dt[i, 0]
        y = dt[i, 1]
        data[0, i] = y**2
        data[1, i] = x*y
        data[2, i] = x
        data[3, i] = y
        data[4, i] = 1

        asnw[0,i] = -x**2

    a = (np.linalg.pinv(data))
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    print(s)
    print(vh.shape)
    print(u.shape)

    print(dt.shape)
    print((asnw@a)[0] @dt)
    return (asnw@a)[0]


def get_ell(dt):
    print(dt)
    a=1
    b=dt[1]
    c=dt[0]
    d=dt[2]
    e=dt[3]
    f=dt[4]

    A = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) * (
                (a + c) + np.sqrt((a - c) ** 2 + b ** 2))) / (b**2-4*a*c)
    B = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) * (
                (a + c) - np.sqrt((a - c) ** 2 + b ** 2))) / (b**2-4*a*c)
    aa=max(A,B)
    bb=min(A,B)
    theta = np.arctan(1/b*(c-a-np.sqrt((a-c)**2+b**2)))
    x0= (2*c*d-b*3)/(b**2-4*a*c)
    y0 = (2*a*e - b*d) / (b**2-4*a*c)
    return aa,bb,x0,y0,theta



def draw(event, x, y, flags, param):
    global x1, y1, click, ptlst,dot_num,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])


        print("점 설정 : (" + str(x1) + ", " + str(y1) + ")")

        if len(ptlst) ==int(dot_num):

            A,B,x0,y0,theta = get_ell(get_circle(np.int32([ptlst])))
            cv2.ellipse(img, (int(x0),int(y0)), (int(A),int(B)) ,theta, 0, 360 , (255, 0, 0) ,2, -1)
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
