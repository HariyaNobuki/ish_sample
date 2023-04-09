import pandas as pd
import numpy as np
import seaborn as sns
import crayons
import time
import os,sys

class Logger:
    def __init__(self,cnf) -> None:
        self.cnf = cnf

    def _log_main_data(self,gen=None,eval=None,fitness=None,test_fitness=None,R2=None,test_R2=None):
        log_list = {"generation":gen,"num_eval":eval,"fitness":fitness,"test_fitness":test_fitness,"R2":R2,"test_R2":test_R2}
        return log_list
    def _log_main_data_save(self,df):
        df_n = pd.DataFrame(df)        # new
        if os.path.isfile(self.cnf.c_respath + "/_log_main.csv"):
            df_o = pd.read_csv(self.cnf.c_respath + "/_log_main.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.c_respath + "/_log_main.csv",index = False)
        else:
            df_n.to_csv(self.cnf.c_respath + "/_log_main.csv",index = False)