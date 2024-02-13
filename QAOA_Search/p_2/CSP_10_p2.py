import numpy as np                                          # 导入numpy库并简写为np
import pandas as pd
import itertools
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from QAOA import _Rx,_Rz,_Rzz,_Rzzz,_Rzzzz,I,_H,getBaseProb

def I(n): #返回可占用n个比特位的I矩阵
    return np.eye(2**n)
z = np.array([[1,0],
                [0,-1]],dtype=np.complex128)
H_C = np.kron(z,I(3)) + np.kron(np.kron(I(2),z),I(1)) - np.kron(I(3),z)\
        - np.kron(np.kron(np.kron(z,z),I(1)),z)

# ------------------p = 2 template 8th-----------------------------
def Qaoa2Stat(gamma_1,beta_1,gamma_2,beta_2,t):
    z = np.array([[1,0],
                    [0,-1]],dtype=np.complex128)
    A = _Rx(0,4,beta_2)*_Rx(1,4,beta_2)*_Rx(2,4,beta_2)*_Rx(3,4,beta_2) \
        *_Rz(0,4,gamma_2)*_Rz(2,4,gamma_2)*_Rz(3,4,-gamma_2) \
        *_Rzzz(0,1,3,4,-gamma_2) \
        *_Rx(0,4,beta_1)*_Rx(1,4,beta_1)*_Rx(2,4,beta_1)*_Rx(3,4,beta_1) \
        *_Rz(0,4,gamma_1)*_Rz(2,4,gamma_1)*_Rz(3,4,-gamma_1) \
        *_Rzzz(0,1,3,4,-gamma_1) \
        *(_H() + _H() + _H() + _H()) #代表量子线路的矩阵
    B = A.dot(t) # |gamma,beta>
    return B
def Qaoa2E(gamma_1,beta_1,gamma_2,beta_2,t,H_C):

    pureState = Qaoa2Stat(gamma_1,beta_1,gamma_2,beta_2,t)
    E = np.conjugate(np.array(pureState)).T.dot(H_C).dot(np.array(pureState ))[0][0].real
    return E

#-------------------读取N参数
parser = argparse.ArgumentParser(description='CSP-问题8-p=2')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--Num', type=int,required=True, help='每个参数遍历的次数')
args = parser.parse_args()
args = vars(args)
print(args)

N = args["Num"]#采样数


t = np.concatenate((np.array([1]) , np.zeros((15)))).reshape(-1,1) #创建初态
beta_1Lst = np.linspace(0,np.pi,N) #生成遍历参数列表
gamma_1Lst = np.linspace(0,2*np.pi,N*2) 
beta_2Lst = np.linspace(0,np.pi,N)
gamma_2Lst = np.linspace(0,2*np.pi,N*2) 
#*********** 并行版本 ******************
loopSet = itertools.product(gamma_1Lst,beta_1Lst,gamma_2Lst,beta_2Lst)
E_lst = Parallel(n_jobs=-1,verbose=0)(delayed(Qaoa2E)(gamma_1,beta_1,gamma_2,beta_2,t,H_C) 
                for gamma_1,beta_1,gamma_2,beta_2 in tqdm(list(loopSet)) )
df = pd.DataFrame(list(itertools.product(gamma_1Lst,beta_1Lst,gamma_2Lst,beta_2Lst)), columns=["gamma_1","beta_1","gamma_2","beta_2"]) #得到参数的数据框

df.insert(0,column="E",value = E_lst) #将E_lst插入数据框
row = df.loc[df["E"].idxmax(),:] # E_max对应列
pureState = Qaoa2Stat(row["gamma_1"],row["beta_1"],row["gamma_2"],row["beta_2"],t) # 返回该最优量子线路输出的纯态
print("QAOA Calculate completed!")
print("最优参数: gamma_1 = {}, beta_1 = {}, gamma_2 = {}, beta_2 = {}".format(row["gamma_1"],row["beta_1"],row["gamma_2"],row["beta_2"]))
print("输出纯态: {}".format(np.around(pureState)))
print("Start ploting")
State = getBaseProb(pureState) # 返回每个基态的概率
Xlabel = ["|{}{}{}{}⟩".format(x,y,z,r) for x,y,z,r in itertools.product([0,1],[0,1],[0,1],[0,1])] # 创建狄拉克符号的基态字符串列表
plt.figure(figsize=(10,6))
plt.bar(Xlabel,State.real)
plt.savefig("p=2_CSP_10.jpg", bbox_inches = "tight")
