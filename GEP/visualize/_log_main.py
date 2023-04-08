"""
for analysis 
PLOT
&
TEST PLOT
"""

import numpy as np
import pandas as pd
import random
import crayons
import os , sys

import ana_quantule

# normal_gep is mistake in naming
#method = ["continuous_gep_RBF","discontinuous_gep_RBF","continuous_gep_KRG","discontinuous_gep_KRG","gep"]
method = ["gep"]
res_path = "../_result/"
#res_path = "//192.168.11.8/Experiment//2023_hariya"+"/SymbolicRegression"+"/"+"Ngnyen-7"+"/"
max_eval = 60

if __name__ == '__main__':
    print(crayons.red("### LOG MAIN ###"))
    df_all = pd.DataFrame()
    df_test_all = pd.DataFrame()
    for method_i in method:
        is_dir = os.path.isdir(res_path + method_i)
        if is_dir:
            print(f"{method_i} is a directory.")
            df_fit = pd.DataFrame()
            df_test_fit = pd.DataFrame()
            folderfile = os.listdir(res_path + method_i)
            for folderfile_i in folderfile:
                if os.path.exists(res_path + method_i + "/"+folderfile_i+"/_log_main.csv"):
                    df_i = pd.read_csv(res_path + method_i + "/" + folderfile_i + "/_log_main.csv")
                    if max(df_i["num_eval"]) < max_eval:
                        break
                    else:   # this cross is right root
                        if df_all.shape == (0, 0):  # init dict
                            df_all["generation"] = df_i["generation"][0:max_eval+1]
                            df_all["num_eval"] = df_i["num_eval"][0:max_eval+1]
                            df_test_all["generation"] = df_i["generation"][0:max_eval+1]
                            df_test_all["num_eval"] = df_i["num_eval"][0:max_eval+1]
                            df_fit["fitness_{}".format(folderfile_i)] = df_i["fitness"][0:max_eval+1]
                            df_test_fit["test_fitness_{}".format(folderfile_i)] = df_i["test_fitness"][0:max_eval+1]
                        else:
                            df_fit["fitness_{}".format(folderfile_i)] = df_i["fitness"][0:max_eval+1]
                            df_test_fit["test_fitness_{}".format(folderfile_i)] = df_i["test_fitness"][0:max_eval+1]
            q1,q2,q3,q4,q5 = ana_quantule.q1_q5(df_fit)
            df_all[method_i+"_q1"] = q1
            df_all[method_i+"_q2"] = q2
            df_all[method_i+"_q3"] = q3
            df_all[method_i+"_q4"] = q4
            df_all[method_i+"_q5"] = q5
            q1,q2,q3,q4,q5 = ana_quantule.q1_q5(df_test_fit)
            df_test_all[method_i+"_q1"] = q1
            df_test_all[method_i+"_q2"] = q2
            df_test_all[method_i+"_q3"] = q3
            df_test_all[method_i+"_q4"] = q4
            df_test_all[method_i+"_q5"] = q5
    ana_quantule.fig_q1_q5(df_all[df_all['num_eval']<max_eval],method,res_path,"train")
    ana_quantule.fig_q2_q4(df_all[df_all['num_eval']<max_eval],method,res_path,"train")
    ana_quantule.fig_q1_q5(df_test_all,method,res_path,"test")
    ana_quantule.fig_q2_q4(df_test_all,method,res_path,"test")


