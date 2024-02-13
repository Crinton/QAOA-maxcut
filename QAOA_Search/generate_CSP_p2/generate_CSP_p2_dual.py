import numpy as np                                          # 导入numpy库并简写为np
import pandas as pd
import itertools
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from QAOA_CSP import CSP, _H,getBaseProb,_Rx,_Rz,_Rzz,_Rzzz,_Rzzzz,I,_H,getBaseProb
from decimal import Decimal
from scipy.optimize import dual_annealing
def create_new_dict(nested_dict, func) -> dict:
    new_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            new_dict[key] = create_new_dict(value, func)
        else:
            new_key, new_value = func(key, value)
            new_dict[new_key] = new_value
    
    return new_dict

# 将二层的key转化为我要的形式的字符串
def modify_key_value(key, value):
    operLst = ",".join(list(map(lambda x:str(eval(x[2]) - 1),key.split())))
    return operLst, value

def Qaoa3E(Problem:dict,HC,gammaLst:list,betaLst:list,p:int,s:np.ndarray) -> float:
    model = CSP(Problem,HC)
    model.updateQC(gammaLst,betaLst,p)
    return model.getExpctation(s)

# Problem转为经典表达式
def paraCSP(Problem,z1,z2,z3,z4):
    C = 0
    Problem_beta = Problem[1]
    Constant = Problem[2]
    factor = Problem[3]
    for _, Power in Problem_beta.items(): #每个次幂的
        for key, coef in Power.items(): #某个次幂中 每个项
            # 解析每个项的表达式，并乘value系数
            item = 1
            for oper in key.split(): 
                item  = item * eval("z{}".format(oper[2]))
            C  = C + item*coef 
    return factor*(C+Constant)
def GridSearch_CSP(Problem):
    sulotionsSet = itertools.product([-1,1],repeat=4)
    obj_maxLst = [0]
    para_max = []
    for z1,z2,z3,z4 in sulotionsSet:
        obj = paraCSP(Problem,z1,z2,z3,z4)
        if obj > obj_maxLst[0]:
            obj_maxLst = [obj]
            para_max = [(z1,z2,z3,z4)]
        elif obj == obj_maxLst[0]:
            obj_maxLst.append(obj)
            para_max.append((z1,z2,z3,z4))
        else:
            pass
    return obj_maxLst,para_max
# 结果评价指标 信息熵

def Entropy(probabilities:np.ndarray):
    # 计算数组中每个元素的概率
    # 计算信息熵
    entropy = -np.sum(probabilities * np.log(probabilities))
    # 归一化信息熵
    n_entropy = 1 - (entropy / np.log(probabilities.size))
    
    return n_entropy
def getParaQC(Problem,gamma:float, beta:float):
    Problem_alpha = Problem[0]
    Problem_beta = Problem[1]
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
def QC_func(Problem, reversed = False): #输出一个func，对应Problem
    model = CSP(Problem)
    HC = model.HC
    def func(X):
        #print(X)
        if X.shape[0] == 4:
            gamma_1,beta_1,gamma_2,beta_2 = X[0],X[1],X[2],X[3]
        elif X.shape[0] ==2:
            gamma_1,beta_1,gamma_2,beta_2 = X[0],X[1],0,0
        else:
            raise ValueError("X 输入不对")
        QC = getParaQC(Problem,gamma_2,beta_2) * getParaQC(Problem,gamma_1,beta_1)
        global s
        rb = QC.dot(s)
        E = (np.conjugate(rb).T.dot(HC).dot(rb))[0][0].real
        if reversed:
            #print(-E)
            return -E
        else:
            return E
    return func
def process_p2(row,i,p): #并行计算p=2时每种问题的情况
    #start = time.time()
    CtrlSeries = row["formula"]
    Power_1 = {"Z_1":CtrlSeries[0], "Z_2":CtrlSeries[1], "Z_3":CtrlSeries[2],"Z_4":CtrlSeries[3]}
    Power_2 = {"Z_1 Z_2":CtrlSeries[4], "Z_3 Z_4":CtrlSeries[5],
            "Z_1 Z_3":CtrlSeries[6], "Z_1 Z_4":CtrlSeries[7], "Z_2 Z_3":CtrlSeries[8], "Z_2 Z_4":CtrlSeries[9]}
    Power_3 = {"Z_1 Z_2 Z_3":CtrlSeries[10], "Z_1 Z_2 Z_4":CtrlSeries[11], "Z_1 Z_3 Z_4": CtrlSeries[12] , "Z_2 Z_3 Z_4":CtrlSeries[13]}
    Power_4 = {"Z_1 Z_2 Z_3 Z_4":CtrlSeries[14]}
    Problem_beta = {"Power_1":Power_1,
            "Power_2":Power_2,
            "Power_3":Power_3,
            "Power_4":Power_4}
    Problem_alpha = create_new_dict(Problem_beta , modify_key_value)
    Problem = (Problem_alpha,Problem_beta, Constant, factor)
    #----------------- 获取CSP(Problem)
    model = CSP(Problem)
    func = QC_func(Problem,True)
    global bounds
    result = dual_annealing(func,bounds,restart_temp_ratio=1e-3,visit = 1.5,maxiter = 1000)
    X = result.x
    E = -result.fun
    model.updateQC([X[0],X[2]],[X[1],X[3]],p)
    pureState = model.getState(s) # 返回该最优量子线路输出的纯态
    State = getBaseProb(pureState).real #得到概率向量
    idxMax = np.argmax(State) #binary变量为 二进制编码的解
    binary = bin(idxMax)[2:].zfill(4)
    #--------------- 上述为量子部分，下述为经典网格搜索部分
    ObjLst,Num_solution_Lst = GridSearch_CSP(Problem)
    New_Num_solution_Lst = []
    for lt in Num_solution_Lst:
        New_Num_solution_Lst.append(''.join(str(int((1-x)/2)) for x in lt))
    #------------输出期望，QAOA的最优解，概率向量的信息熵，gamma_1,gamma_2, beta_1,beta_2, 经典最优目标值向量(可能由多解)，经典最优解向量，QAOA的解是否为最优解之一
    #end = time.time()
    #print(i, " Time cost is minutes", np.round((end-start)/60,4))
    return Decimal(E),binary,Entropy(State),X[0],X[2],X[1],X[3],ObjLst,New_Num_solution_Lst,binary in New_Num_solution_Lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSP-问题8-p=2')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('--Num', type=int,required=True, help='每个参数遍历的次数')
    parser.add_argument('--jobs', type=int,required=True, help='核心数')

    args = parser.parse_args()
    args = vars(args)
    print(args)
    N = args["Num"]#采样数
    num_cores = args["jobs"]
    External = 1
    p = 2 # p = 1

    bounds = [[0,2*np.pi],
            [0,np.pi],
            [0,2*np.pi],
            [0,np.pi]]
    combinations = itertools.product([0, -1, 1], repeat=15)
    CtrlSeries_Lst = [c for c in combinations if sum(1 for i in c[6:] if i != 0) == External]
    df_CSP = pd.DataFrame(data = {"E":np.nan,
                                "solutions":np.nan,
                                "real_solutions":np.nan,
                                "isOpt":np.nan,
                                "Entropy":np.nan,
                                "gamma_1":np.nan,
                                "gamma_2":np.nan,
                                "beta_1":np.nan,
                                "beta_2":np.nan,
                                "formula":CtrlSeries_Lst})
    Constant = 0
    factor = 1

    t = np.concatenate((np.array([1]) , np.zeros((15)))).reshape(-1,1) #创建初态
    s = (_H() + _H() + _H() + _H()).dot(t).to_array()
    gamma_1Lst = np.linspace(0,2*np.pi,N*2) #生成遍历参数列表
    beta_1Lst = np.linspace(0,np.pi,N) 

    loopSeries = list(itertools.product(gamma_1Lst,beta_1Lst))
    fill_data = Parallel(n_jobs=num_cores,timeout = None)(delayed(process_p2)(row,i,p) 
                        for i,row in tqdm(list(df_CSP.iterrows())))
    fill_df = pd.DataFrame(data = fill_data,columns = ["E", "solutions", "Entropy","gamma_1","gamma_2", "beta_1", "beta_2", "MaxObj","real_solutions","isOpt"])
    for col in fill_df: #将输出填入df_CSP数据框中
        df_CSP[col] = fill_df[col]
    tmpLst = df_CSP.apply(lambda row: row["E"]/ row["MaxObj"][0],axis = 1)
    df_CSP.insert(1,"ratio",value=tmpLst)
    print("计算完成")
    df_CSP.round(8).to_csv("df_CSP_p1_External={}_Deci.csv".format(External),encoding = "utf-8")