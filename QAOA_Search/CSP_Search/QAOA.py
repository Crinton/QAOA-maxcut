from scipy.linalg import expm
import numpy as np                                          
class PauliOperator(np.ndarray):
    """
    只实现了重写"+"为kron tensor, 懒得每次np.kron
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    def __add__(self, other):
        return np.kron(self, other)
    def __mul__(self, other):
        return np.dot(self, other)
    def __eq__(self, other):
        return np.array_equal(self, other)
    def to_array(self):
        return np.array(self)
def _Rx(obj_b:int,ndim:int, theta:float) -> PauliOperator:
    x = np.array([[0,1],
            [1,0]],dtype=np.complex128)  
    theta = np.mod(theta,2*np.pi)  
    if obj_b == 0:
        mat_1 = PauliOperator(1)
    else:
        mat_1 = PauliOperator(np.eye(2**obj_b,dtype=np.complex128))
    mat_2 = PauliOperator(x)
    mat_3 = PauliOperator(np.eye(2**(ndim-obj_b-1),dtype=np.complex128))
    mat = mat_1 + mat_2 + mat_3 # 连续np.kron
    return PauliOperator(expm(-1j*theta/2 * mat))
def _Rz(obj_b:int,ndim:int, theta:float) -> PauliOperator:
    z = np.array([[1,0],
            [0,-1]],dtype=np.complex128)    
    theta = np.mod(theta,2*np.pi)
    if obj_b == 0:
        mat_1 = PauliOperator(1)
    else:
        mat_1 = PauliOperator(np.eye(2**obj_b,dtype=np.complex128))
    mat_2 = PauliOperator(z)
    mat_3 = PauliOperator(np.eye(2**(ndim-obj_b-1),dtype=np.complex128))
    mat = mat_1 + mat_2 + mat_3 # 连续np.kron
    return PauliOperator(expm(-1j*theta/2 * mat))
def _Rzz(ctro_b:int, obj_b:int, ndim:int, theta:float) -> PauliOperator:
    z = PauliOperator(np.array([[1,0],
            [0,-1]],dtype=np.complex128))
    theta = np.mod(theta,2*np.pi)
    if ctro_b == 0:
        mat_1 = PauliOperator(1)
    else:
        mat_1 = PauliOperator(np.eye(2**ctro_b,dtype=np.complex128))

    mat_2 = z +  PauliOperator(np.eye(2**(obj_b - ctro_b-1),dtype=np.complex128)) + z
    mat_3 = PauliOperator(np.eye(2**(ndim-obj_b-1),dtype=np.complex128))
    mat = mat_1 + mat_2 + mat_3
    return PauliOperator(expm(-1j*theta/2 * mat))


def _Rzzz(qubit_1:int, qubit_2:int, qubit_3:int, ndim:int, theta:float) -> PauliOperator:
    z = np.array([[1,0],
            [0,-1]],dtype=np.complex128)
    theta = np.mod(theta,2*np.pi)
    if qubit_1 == 0:
            mat_1 = PauliOperator(1)
    else:
        mat_1 = PauliOperator(np.eye(2**qubit_1,dtype=np.complex128))
    mat_2 = PauliOperator(z.copy()) #qubit_1的z
    mat_3 = PauliOperator(np.eye(2**(qubit_2 - qubit_1-1),dtype=np.complex128))
    mat_4 = PauliOperator(z.copy()) #qubit_2的z
    mat_5 = PauliOperator(np.eye(2**(qubit_3 - qubit_2-1),dtype=np.complex128))
    mat_6 = PauliOperator(z.copy()) #qubit_3的z
    mat_7 = PauliOperator(np.eye(2**(ndim-qubit_3-1),dtype=np.complex128))
    mat = mat_1 + mat_2 + mat_3 + mat_4 + mat_5 + mat_6 + mat_7
    return PauliOperator(expm(-1j*theta/2 * mat))
def _Rzzzz(theta:float) -> PauliOperator:
    z = PauliOperator(np.array([[1,0],
                                [0,-1]],dtype=np.complex128))
    theta = np.mod(theta,2*np.pi)
    return PauliOperator(expm(-1j*theta/2 * (z + z + z + z)))
def _H():
    return 1/np.sqrt(2) * PauliOperator(np.array([[1,1],
                                                  [1,-1]],dtype=np.complex128))
def I(n): #返回可占用n个比特位的I矩阵
    return np.eye(2**n)
def getBaseProb(psi):
    # 计算密度矩阵
    rho = np.outer(psi, psi.conj())
    # 返回各个基态的概率
    return np.diag(rho)