import optparse
import pandas as pd

usage = "python execute_descriptive.py --file FILE_NAME --columns col1,col2 --method [fisher,pearson,paired] --value FLOAT"
parser = optparse.OptionParser(usage, version="%prog 1.0")
parser.add_option("--file", dest="filename",type="string",metavar="FILE_NAME",help="待处理的CSV文件")
parser.add_option("--method",type="string",help="卡方检验的方法")
#parser.add_option("--value",type="float",help="T检")
parser.add_option("--columns", action="store",type="string",help="需要检验的列名列表") 
parser.add_option("--fexp", action="store",type="string",help="专用于拟合优度检验的期望频数") 

options, args = parser.parse_args() #options是正常参数, args是未能识别的参数
if len(args) != 0:
    raise KeyError("输入格式错误，有参数输入未被捕获")
print("options = {}, args = {}".format(options, args))
args = vars(options)
print(args)