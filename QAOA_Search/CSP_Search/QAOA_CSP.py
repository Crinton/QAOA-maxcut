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



class CSP(object):
    def __init__(self,Problem:dict,HC:np.ndarray = None):
        self.Problem = Problem
        if HC is not None: 
            self.HC = HC
        else:
            self.HC = self.getHamiltonianOper()
    def _Z(self,obj_b:int,ndim:int) -> PauliOperator:
        z = np.array([[1,0],
                [0,-1]],dtype=np.complex128)    
        if obj_b == 0:
            mat_1 = PauliOperator(1)
        else:
            mat_1 = PauliOperator(np.eye(2**obj_b,dtype=np.complex128))
        mat_2 = PauliOperator(z)
        mat_3 = PauliOperator(np.eye(2**(ndim-obj_b-1),dtype=np.complex128))
        return mat_1 + mat_2 + mat_3 # 连续np.kron
    
    def getOper(self,string:str, coef:int) -> PauliOperator:
        '''
        Desc: 输入一个字符串形式的泡利算符,返回扩充相应维度并且乘上符号(coef)的泡利算符
        Input: string, str, 泡利算符字符串，如 "Z_1" "Z_1 Z_2", "Z_1 Z_2 Z_3 Z_4"
            coef, int, 该算符相应的符号 {-1,1}
        Output: Oper, PauliOperator
        '''    
        
        operLst = string.split()
        Ham = 1
        for oper in operLst:
            Ham *= self._Z(eval(oper[-1])-1,4)
        return coef*Ham
    
    def getHamiltonianOper(self) -> np.array:
        '''
        调用了getOper函数
        '''

        Problem_beta = self.Problem[1]
        Constant = self.Problem[2]
        factor = self.Problem[3]

        Power_oper = 0
        for _,item_Power in Problem_beta.items(): #计算H_C
            for string_Oper,coef in item_Power.items():
                Power_oper +=self.getOper(string_Oper,coef).to_array() # 该函数已经乘符号, 直接相加(矩阵相加, 不是tensor, 已经to_array().
        H_C = factor*(I(4)*Constant + Power_oper)
        return H_C
    
    def getParaQC(self, gamma:float, beta:float) -> PauliOperator:
        Problem_alpha = self.Problem[0]
        Problem_beta = self.Problem[1]
        QC = 1
        for (qubit_str, _), (_,coef) in zip(Problem_alpha["Power_1"].items(),Problem_beta["Power_1"].items()):
            if coef != 0:
                QC = _Rz(eval(qubit_str),4,coef*gamma) * QC
        for (qubit_str, _), (_,coef) in zip(Problem_alpha["Power_2"].items(),Problem_beta["Power_2"].items()):
            if coef != 0:
                qubit_idx = eval(qubit_str)
                QC = _Rzz(qubit_idx[0],qubit_idx[1],4,coef*gamma) * QC
        for (qubit_str, _), (_,coef) in zip(Problem_alpha["Power_3"].items(),Problem_beta["Power_3"].items()):
            if coef != 0:
                qubit_idx = eval(qubit_str)
                QC = _Rzzz(qubit_idx[0],qubit_idx[1],qubit_idx[2],4,coef*gamma) * QC

        if Problem_alpha["Power_4"]['0,1,2,3'] != 0:
            QC = _Rzzzz(Problem_alpha["Power_4"]['0,1,2,3']*gamma) *  QC
        QC = _Rx(0,4,beta)*_Rx(1,4,beta)*_Rx(2,4,beta)*_Rx(3,4,beta) * QC
        return QC
    def updateQC(self, gammaLst:list, betaLst:list, p:int) -> None:
        '''
        调用了getParaRz函数
        生成关于Problem_beta字典 gamma与beta参数的量子线路等价矩阵, 该矩阵还需要在外层附上H_gate。
        '''
        QC = 1
        for layer in range(p):
            QC = self.getParaQC(gammaLst[layer],betaLst[layer])  * QC 
        #QC = QC * (_H() + _H() + _H() + _H()) 
        # H门不在这里作用，提前作用好
        self.QC = QC
    def getState(self,t):
        B = self.QC.dot(t) # |gamma,beta_2>
        return B
    def getExpctation(self,t):
        pureState = self.getState(t)
        E = np.conjugate(pureState).T.dot(self.HC).dot(pureState)[0][0].real
        return E