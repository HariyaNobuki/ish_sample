"""
graderion
"""

import numpy as np
import pandas as pd
import random
import crayons
import os , sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import ana_quantule

# normal_gep is mistake in naming
method = ["continuous_gep_RBF","discontinuous_gep_RBF","continuous_gep_KRG","discontinuous_gep_KRG","gep"]
#method = ["discontinuous_gep_RBF","continuous_gep_RBF","gep"]
#res_path = "_result/"
res_path = "//192.168.11.8/Experiment//2023_hariya"+"/SymbolicRegression"+"/"+"Keijizer11"+"/"
max_eval = 1000

def plotgraph(df):
    plt.plot(df["local_i"],df["f_sur"],lw=0.5,ms=1.0,color=cm.Blues(min(df["num_eval"])/(min_eval+50)))

def customgraph():
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
                if os.path.exists(res_path + method_i + "/"+folderfile_i+"/_log_local_refinement_best.csv"):
                    df_i = pd.read_csv(res_path + method_i + "/" + folderfile_i + "/_log_local_refinement_best.csv")
                    min_eval = min(df_i["num_eval"])
                    max_eval = max(df_i["num_eval"])
                    fig = plt.figure(figsize = (6,4))
                    for eval in range(min_eval,max_eval):
                        if df_i[df_i['num_eval']==eval].shape[0] != 0:
                            plotgraph(df_i[df_i['num_eval']==eval])
                    customgraph()
                    fig.savefig(res_path + method_i + "/"+folderfile_i+"/_log_local_refinement_best.png")




