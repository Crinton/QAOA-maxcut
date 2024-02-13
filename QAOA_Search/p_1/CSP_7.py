import numpy as np                                          
import pandas as pd
from QAOA import _Rx,_Rz,_Rzz,_Rzzz,_Rzzzz,I,_H,getBaseProb
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt


def QaoaGetPara(gamma,beta,t,isOutStat = False):
    z = np.array([[1,0],
                    [0,-1]],dtype=np.complex128)
    A = _Rx(0,4,beta)*_Rx(1,4,beta)*_Rx(2,4,beta)*_Rx(3,4,beta) \
        *_Rz(0,4,gamma)*_Rz(1,4,gamma)*_Rz(2,4,gamma)*_Rz(3,4,gamma) \
        *(_Rzz(0,1,4,-gamma))*(_Rzz(2,3,4,-gamma))*_Rzzz(0,1,2,4,gamma)*_Rzzzz(gamma) \
        *(_H() + _H() + _H() + _H()) #代表量子线路的矩阵
    B = A.dot(t) # |gamma,beta>
    if isOutStat:
        return B
    H_C = np.kron(z,I(3)) + np.kron(np.kron(I(1),z),I(2)) + np.kron(np.kron(I(2),z),I(1)) +  \
          np.kron(I(3),z) - np.kron(np.kron(z,z),I(2)) - np.kron(I(2),np.kron(z,z)) + \
          np.kron(np.kron(np.kron(z,z),z),I(1)) + np.kron(np.kron(np.kron(z,z),z),z)
    E = np.conjugate(np.array(B)).T.dot(H_C).dot(np.array(B))[0][0].real # 计算期望
    return (E,gamma,beta)

if __name__ == "__main__":
    t = np.concatenate((np.array([1]) , np.zeros((15)))).reshape(-1,1)
    beta_lst = np.linspace(0,np.pi,50)
    gamma_lst = np.linspace(0,2*np.pi,100) 
    #*********** 串行版本 ******************
    table = Parallel(n_jobs=14)(delayed(QaoaGetPara)(gamma,beta,t) for gamma,beta in itertools.product(gamma_lst,beta_lst)) #并行版本
    table = pd.DataFrame(table,columns = ["E","gamma","beta"])
    row_idx = table["E"].idxmax()
    row = table.loc[row_idx,["gamma", "beta"]] # E_max对应列
    pureState = QaoaGetPara(row["gamma"],row["beta"],t,True) # 返回该最优量子线路输出的纯态
    print("QAOA Calculate completed!")
    print("最优参数: gamma = {}, beta = {}".format(row["gamma"],row["beta"]))
    print("输出纯态: {}".format(pureState))
    print("Start ploting")
    # 绘制热力图
    matrix = table.round(4).pivot(index='gamma', columns='beta', values='E')
    sns.heatmap(matrix, cmap='coolwarm', fmt='.0f')
    plt.savefig("heatmap-CSP_7.jpg", bbox_inches = "tight")
    # 创建狄拉克符号的基态字符串列表
    Xlabel = ["|{}{}{}{}⟩".format(x,y,z,r) for x,y,z,r in itertools.product([0,1],[0,1],[0,1],[0,1])]
    plt.figure(figsize=(10,6))
    State = getBaseProb(pureState) # 返回每个基态的概率
    plt.bar(Xlabel,State.real)
    plt.savefig("bar-CSP_7.jpg", bbox_inches = "tight")

    print("heatmap bar output finished!")