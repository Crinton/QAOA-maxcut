import numpy as np                                          # 导入numpy库并简写为np
import pandas as pd
import itertools
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from QAOA_CSP import CSP, _H,getBaseProb
import time
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

def process_p2(row,i,p): #并行计算p=2时每种问题的情况
    start = time.time()
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
    HC = model.HC
    #loopSeries = list(itertools.product(gamma_1Lst,gamma_1Lst,beta_1Lst,beta_1Lst))
    E_lst = Parallel(n_jobs=1,verbose=0)(delayed(Qaoa3E)(Problem,HC,[gamma_1,gamma_2],[beta_1,beta_2],p,s) 
                    for gamma_1,gamma_2,beta_1,beta_2 in loopSeries)
    df = pd.DataFrame(loopSeries, columns=["gamma_1","gamma_2","beta_1","beta_2"]) #得到参数的数据框
    df.insert(0,column="E",value = E_lst) #将E_lst插入数据框
    row = df.loc[df["E"].idxmax(),:] # E_max对应列
    model.updateQC([row["gamma_1"],row["gamma_2"]],[row["beta_1"],row["beta_2"]],p)
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
    end = time.time()
    print(i, " Time cost is minutes", np.round((end-start)/60,4))
    return row["E"],binary,Entropy(State),row["gamma_1"],row["gamma_2"],row["beta_1"],row["beta_2"],ObjLst,New_Num_solution_Lst,binary in New_Num_solution_Lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSP-问题8-p=2')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('--Num', type=int,required=True, help='每个参数遍历的次数')
    parser.add_argument('--jobs', type=int,required=True, help='核心数')
    parser.add_argument('--External', type=int,required=True, help='外部项的个数')

    args = parser.parse_args()
    args = vars(args)
    print(args)
    N = args["Num"]#采样数
    num_cores = args["jobs"]
    External = args["External"]
    p = 2 # p = 1


    combinations = itertools.product([0, -1, 1], repeat=15)
    CtrlSeries_Lst = [c for c in combinations if sum(1 for i in c[6:] if i != 0) == External][:200]
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

    loopSeries = list(itertools.product(gamma_1Lst,gamma_1Lst,beta_1Lst,beta_1Lst))
    fill_data = Parallel(n_jobs=num_cores,timeout = None)(delayed(process_p2)(row,i,p) 
                        for i,row in tqdm(list(df_CSP.iterrows())))
    fill_df = pd.DataFrame(data = fill_data,columns = ["E", "solutions", "Entropy","gamma_1","gamma_2", "beta_1", "beta_2", "MaxObj","real_solutions","isOpt"])
    for col in fill_df: #将输出填入df_CSP数据框中
        df_CSP[col] = fill_df[col]
    tmpLst = df_CSP.apply(lambda row: row["E"]/ row["MaxObj"][0],axis = 1)
    df_CSP.insert(1,"ratio",value=tmpLst)
    print("计算完成")
    df_CSP.round(8).to_csv("df_CSP_p1_External={}.csv".format(External),encoding = "utf-8")


