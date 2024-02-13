import sys
import os
import shutil
#os.remove(path) 
base_path = os.getcwd()
source_path = base_path + "/QAOA.py"
for filename in os.listdir("./"):
    if os.path.isdir(filename):
        target_path = base_path + "/"+filename + "/QAOA.py"
        if os.path.isfile(target_path):
            print(target_path)
           #os.remove(target_path) 
            shutil.copy(source_path, target_path)
        else:
            print(target_path)
            shutil.copy(source_path, target_path)