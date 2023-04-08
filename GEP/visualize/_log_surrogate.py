"""
for analysis 
SURROGATE
"""

import numpy as np
import pandas as pd
import random
import crayons
import os , sys

import ana_quantule

# normal_gep is mistake in naming
method = ["continuous_gep_RBF","discontinuous_gep_RBF","continuous_gep_KRG","discontinuous_gep_KRG"]
#method = ["discontinuous_gep_RBF","continuous_gep_RBF","normal_gep_RBF"]
res_path = "//192.168.11.8/Experiment//2023_hariya"+"/SymbolicRegression"+"/"+"Ngnyen-7"+"/"
#res_path = "_result/"
max_eval = 1000

if __name__ == '__main__':
    print(crayons.red("### LOG SURROGATE ###"))
    df_k_all = pd.DataFrame()
    df_rd_all = pd.DataFrame()
    df_rmse_all = pd.DataFrame()
    for method_i in method:
        is_dir = os.path.isdir(res_path + method_i)
        if is_dir:
            print(f"{method_i} is a directory.")
            df_k = pd.DataFrame()
            df_rd = pd.DataFrame()
            df_rmse = pd.DataFrame()
            folderfile = os.listdir(res_path + method_i)
            for folderfile_i in folderfile:
                if os.path.exists(res_path + method_i + "/"+folderfile_i+"/_log_surrogate.csv"):
                    df_i = pd.read_csv(res_path + method_i + "/" + folderfile_i + "/_log_surrogate.csv")
                    if max(df_i["num_eval"]) < max_eval:
                        break
                    else:   # this cross is right root
                        if df_k_all.shape == (0, 0):  # init dict
                            df_k_all["generation"] = df_i["generation"][0:max_eval+1]
                            df_k_all["num_eval"] = df_i["num_eval"][0:max_eval+1]
                            df_rd_all["generation"] = df_i["generation"][0:max_eval+1]
                            df_rd_all["num_eval"] = df_i["num_eval"][0:max_eval+1]
                            df_rmse_all["generation"] = df_i["generation"][0:max_eval+1]
                            df_rmse_all["num_eval"] = df_i["num_eval"][0:max_eval+1]
                            df_k["kendalltau_{}".format(folderfile_i)] = df_i["kendalltau"][0:max_eval+1]
                            df_rd["rank_dif_{}".format(folderfile_i)] = df_i["rank_dif"][0:max_eval+1]
                            df_rmse["rmse_{}".format(folderfile_i)] = df_i["rmse"][0:max_eval+1]
                        else:
                            df_k["kendalltau_{}".format(folderfile_i)] = df_i["kendalltau"][0:max_eval+1]
                            df_rd["rank_dif_{}".format(folderfile_i)] = df_i["rank_dif"][0:max_eval+1]
                            df_rmse["rmse_{}".format(folderfile_i)] = df_i["rmse"][0:max_eval+1]
            q1,q2,q3,q4,q5 = ana_quantule.q1_q5(df_k)
            df_k_all[method_i+"_q1"] = q1
            df_k_all[method_i+"_q2"] = q2
            df_k_all[method_i+"_q3"] = q3
            df_k_all[method_i+"_q4"] = q4
            df_k_all[method_i+"_q5"] = q5
            q1,q2,q3,q4,q5 = ana_quantule.q1_q5(df_rd)
            df_rd_all[method_i+"_q1"] = q1
            df_rd_all[method_i+"_q2"] = q2
            df_rd_all[method_i+"_q3"] = q3
            df_rd_all[method_i+"_q4"] = q4
            df_rd_all[method_i+"_q5"] = q5
            q1,q2,q3,q4,q5 = ana_quantule.q1_q5(df_rmse)
            df_rmse_all[method_i+"_q1"] = q1
            df_rmse_all[method_i+"_q2"] = q2
            df_rmse_all[method_i+"_q3"] = q3
            df_rmse_all[method_i+"_q4"] = q4
            df_rmse_all[method_i+"_q5"] = q5
    ana_quantule.fig_q1_q5(df_k_all[df_k_all['num_eval']<max_eval],method,res_path,"kendallltau")
    ana_quantule.fig_q2_q4(df_k_all[df_k_all['num_eval']<max_eval],method,res_path,"kendallltau")
    ana_quantule.fig_q1_q5(df_rd_all,method,res_path,"rank_dif")
    ana_quantule.fig_q2_q4(df_rd_all,method,res_path,"rank_dif")
    ana_quantule.fig_q1_q5(df_rmse_all,method,res_path,"rmse")
    ana_quantule.fig_q2_q4(df_rmse_all,method,res_path,"rmse")


