import os , sys
import crayons 

def MakeFiles(filename,path):
    print(crayons.blue("---making files"))
    os.makedirs(path+"/"+filename,exist_ok=True)