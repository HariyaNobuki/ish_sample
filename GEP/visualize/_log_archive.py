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
import matplotlib.pyplot as plt

import ana_quantule

# normal_gep is mistake in naming
method = ["continuous_gep_RBF","discontinuous_gep_RBF","continuous_gep_KRG","discontinuous_gep_KRG","gep"]
#method = ["discontinuous_gep_RBF","continuous_gep_RBF","gep"]
#res_path = "_result/"
res_path = "//192.168.11.8/Experiment//2023_hariya"+"/SymbolicRegression"+"/"+"Ngnyen-7"+"/"
max_eval = 1000

def plotfigure():
    plt.plot(eval_list,q3_list,lw=0.5,ms=1.0,color="red")
    #plt.fill_between(eval_list, q1_list, q5_list,color="red",alpha=0.2)
    plt.fill_between(eval_list, q2_list, q4_list,color="red",alpha=0.3)
    plt.xlabel("evaluation")
    plt.ylabel("obj")


if __name__ == '__main__':
    print(crayons.red("### LOG MAIN ###"))
    df_all = pd.DataFrame()
    for method_i in method:
        is_dir = os.path.isdir(res_path + method_i)
        if is_dir:
            print(f"{method_i} is a directory.")
            df_fit = pd.DataFrame()
            df_test_fit = pd.DataFrame()
            folderfile = os.listdir(res_path + method_i)
            for folderfile_i in folderfile:
                if os.path.exists(res_path + method_i + "/"+folderfile_i+"/_log_archive.csv"):
                    q1_list,q2_list,q3_list,q4_list,q5_list = [],[],[],[],[]
                    eval_list = []
                    df_i = pd.read_csv(res_path + method_i + "/" + folderfile_i + "/_log_archive.csv")
                    min_eval = min(df_i["eval"])
                    max_eval = max(df_i["eval"])
                    fig = plt.figure(figsize = (6,4))
                    for eval in range(min_eval,max_eval):
                        if df_i[df_i['eval']==eval].shape[0] != 0:
                            q1,q2,q3,q4,q5 = ana_quantule.q1_q5_axis0(df_i[df_i['eval']==eval])
                            eval_list.append(eval)
                            q1_list.append(q1["obj"])
                            q2_list.append(q2["obj"])
                            q3_list.append(q3["obj"])
                            q4_list.append(q4["obj"])
                            q5_list.append(q5["obj"])
                    plotfigure()
                    fig.savefig(res_path + method_i + "/"+folderfile_i+"/_log_archive.png")
                    plt.close()
                    plt.clf()



