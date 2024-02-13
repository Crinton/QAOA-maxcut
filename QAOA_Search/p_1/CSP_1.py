import numpy as np                                          
import pandas as pd
from QAOA import _Rx,_Rz,_Rzz,I,_H,getBaseProb
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt


def QaoaGetPara(gamma,beta,t,isOutStat = False):
    z = np.array([[1,0],
                    [0,-1]],dtype=np.complex128)
    A = _Rx(0,4,beta)*_Rx(1,4,beta)*_Rx(2,4,beta)*_Rx(3,4,beta) \
        *(-_Rz(0,4,gamma))*_Rz(1,4,gamma)*_Rz(2,4,gamma)*_Rz(3,4,gamma) \
        *_Rzz(0,1,4,gamma)*(-_Rzz(2,3,4,gamma)) \
        *(_H() + _H() + _H() + _H()) #代表量子线路的矩阵
    B = A.dot(t) # |gamma,beta>
    if isOutStat:
        return B
    H_C = -np.kron(z,I(3)) + np.kron(np.kron(I(1),z),I(2)) + np.kron(np.kron(I(2),z),I(1))  \
          +np.kron(I(3),z) + np.kron(np.kron(z,z),I(2)) - np.kron(I(2),np.kron(z,z))
    E = np.conjugate(np.array(B)).T.dot(H_C).dot(np.array(B))[0][0].real # 计算期望
    return (E,gamma,beta)


if __name__ == "__main__":
    t = np.concatenate((np.array([1]) , np.zeros((15)))).reshape(-1,1)
    beta_lst = np.linspace(0,np.pi,50)
    gamma_lst = np.linspace(0,2*np.pi,100) 
    #*********** 串行版本 ******************
    table = Parallel(n_jobs=-1)(delayed(QaoaGetPara)(beta,gamma,t) for beta,gamma in itertools.product(beta_lst,gamma_lst)) #并行版本
    table = pd.DataFrame(table,columns = ["E","gamma","beta"])
    row_idx = table["E"].idxmax()
    row = table.loc[row_idx,["gamma", "beta"]] # E_max对应列
    pureState = QaoaGetPara(row["gamma"],row["beta"],t,True) # 返回该最优量子线路输出的纯态
    State = getBaseProb(pureState) # 返回每个基态的概率
    print("QAOA Calculate completed!")
    print("Start ploting")
    # 绘制热力图
    matrix = table.round(4).pivot(index='gamma', columns='beta', values='E')
    sns.heatmap(matrix, cmap='coolwarm', fmt='.0f')
    plt.savefig("heatmap-CSP_1.jpg", bbox_inches = "tight")
    # 创建狄拉克符号的基态字符串列表
    Xlabel = ["|{}{}{}{}⟩".format(x,y,z,r) for x,y,z,r in itertools.product([0,1],[0,1],[0,1],[0,1])]
    plt.figure(figsize=(10,6))
    plt.bar(Xlabel,State.real)
    plt.savefig("bar-CSP_1.jpg", bbox_inches = "tight")

    print("heatmap bar output finished!")