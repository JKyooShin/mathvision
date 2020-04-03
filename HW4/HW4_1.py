import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
click = False
x1, y1 = -1, -1
def lstmin(a,b):
    return [a[0]-b[0],a[1]-b[1]]
def outer(a,b,c):
    A = lstmin(b,a)
    B=lstmin(c,a)

    S = A[0]*B[1]-A[1]*B[0]
    return (S)



def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click, ptlst,dot_num,end  # 전역변수 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        ptlst.append([x1,y1])


        print("점 설정 : (" + str(x1) + ", " + str(y1) + ")")

        if len(ptlst) == int(dot_num):

            volume=0
            for i in range(1,len(ptlst)-1):
                volume+=(outer(ptlst[0], ptlst[i], ptlst[i+1]))/2
            print(abs(volume))
            ptlst= np.array(ptlst)

            cv2.polylines(img, np.int32([ptlst]),True, (255,0,0), 2)
            ptlst=[]
            end = True
            cv2.putText(img, str(abs(volume)), (100,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,255,0), 2 )




dot_num =input("write in number of dots")


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
