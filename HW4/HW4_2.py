import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
click = False
x1, y1 = -1, -1

def lstmin(a,b):
    return [a[0]-b[0],a[1]-b[1]]
def outer(a,b,c):
    A = lstmin(b,a)
    B=lstmin(c,b)
    S = A[0]*B[1]-A[1]*B[0]
    return (S)

def rec_outer(ptlst):
    outer_lst = []
    outer_lst.append(outer(ptlst[0],ptlst[1],ptlst[2]))
    outer_lst.append(outer(ptlst[1], ptlst[2], ptlst[3]))
    outer_lst.append(outer(ptlst[2], ptlst[3], ptlst[0]))
    outer_lst.append(outer(ptlst[3], ptlst[0], ptlst[1]))
    return outer_lst

def check(ptlst, newlst):
    checker=0
    for i in range(len(ptlst)):
        if ptlst[i]*newlst[i]<0:
            checker+=1
        else:
            checker+=0
    return checker


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click, ptlst,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])

        if len(ptlst)%4==0:
            if len(ptlst) == 4:
                #first_lst = rec_outer(ptlst)
                print("original")
                cv2.polylines(img, np.int32([ptlst]), True, (255, 0, 0), 2)
            else:
                out_lst = rec_outer(ptlst[-4:])
                first_lst = rec_outer(ptlst[:4])
                classifier = check(first_lst, out_lst)
                print("{}st Try".format(int((len(ptlst)-4)/4)))
                if classifier ==0:
                    print("normal")
                elif classifier==1:
                    print("concave")
                elif classifier ==2:
                    print("twist")
                elif classifier == 3:
                    print("Reflected Concave")
                elif classifier == 4:
                    print("Reflect")
                cv2.polylines(img, np.int32([ptlst[-4:]]), True, (255, 0, 255), 2)



img = np.zeros((500, 500, 3), np.uint8)
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
