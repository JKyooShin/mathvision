import numpy as np


class rigid():

    def __init__(self, a,b):
        #a와 b는 기존의 3점과 변하는 3점
        self.a = a
        self.b = b

        #두 벡터를 크기가 같게 만들어준다.
        self.A = self.norm(self.cross(self.a))
        self.B = self.norm(self.cross(self.b))
        #이후 첫 변환, 법선벡터 h에 대해서 h' 으로 옮겨가는 회전 R1을 구한다.
        #이때 회전축은 A와 B의 단위벡터가 된다.
        self.R1 =self.getRigid(self.A, self.B, self.norm(np.cross(self.A, self.B)))
        #h'인 B에 대해 R(p1, p3) 과 p1', p3' 이 일치하도록 회전하는 R2를 구한다.
        self.R2=self.getRigid(np.matmul(self.R1, self.a[2] - self.a[0]), self.b[2] - self.b[0], self.B)
        #이후, 이에 대한 식을 함수로 고정한다.
        #rigid function

    def cross(self, a):
        return np.cross(a[1] - a[0], a[2] - a[0])
    def norm(self,u):
        return u/np.linalg.norm(u)
    def v(self, v):
        return (self.cross(v))

    def Find_cos(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def configure_R(self, cs, u):
        sn = np.sqrt(1-cs**2)
        ux = u[0]
        uy = u[1]
        uz = u[2]
        return [
            [cs + ux ** 2 * (1 - cs), ux * uy * (1 - cs) - uz * sn, ux * uz * (1 - cs) + uy * sn],
            [uy * ux * (1 - cs) + uz * sn, cs + uy ** 2 * (1 - cs), uy * uz * (1 - cs) - ux * sn],
            [uz * ux * (1 - cs) - uy * sn, uz * uy * (1 - cs) + ux * sn, cs + uz ** 2 * (1 - cs)]
        ]
    def getRigid(self, A, B, U):
        R=self.configure_R(self.Find_cos(A, B), U)
        return(R)

    def rigid_fucntion(self, X):
        return np.matmul(np.matmul(self.R1,(X - self.a[0])), self.R2) + self.b[0]
#class 를 생성할 때, 기존 3점 // 변한 3점을 입력한다.
rg = rigid(a=np.array([[-0.5,0,2.121320], [0.5,0,2.121320], [0.5,-0.707107,2.828427]]), b=np.array([[1.363005,-0.427130,2.339082], [1.748084,0.437983,2.017688], [2.636461,0.184843,2.400710]]))
#Class init 시 R1, R2 회전이 계산된다.
print(rg.R1)
print(rg.R2)
#Class 의 Rigid_funcion 함수로 새 좌표를 변환시킬 수 있다.
print(rg.rigid_fucntion(np.array([0.500000, 0.707107, 2.828427])))
print(rg.rigid_fucntion(np.array([1,1,1])))
