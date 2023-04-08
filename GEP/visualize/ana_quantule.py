import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def q1_q5(df):
    q1 = df.quantile(0,axis=1)
    q2 = df.quantile(0.25,axis=1)
    q3 = df.quantile(0.5,axis=1)
    q4 = df.quantile(0.75,axis=1)
    q5 = df.quantile(1.0,axis=1)
    return q1,q2,q3,q4,q5

def q1_q5_axis0(df):
    q1 = df.quantile(0,axis=0)
    q2 = df.quantile(0.25,axis=0)
    q3 = df.quantile(0.5,axis=0)
    q4 = df.quantile(0.75,axis=0)
    q5 = df.quantile(1.0,axis=0)
    return q1,q2,q3,q4,q5

def fig_q1_q5(df_all,method,res_path,name):
    c = {"continuous_gep_RBF":"red",
        "discontinuous_gep_RBF":"blue",
        "continuous_gep_KRG":"green",
        "discontinuous_gep_KRG":"cyan",
        "gep":"yellow"}
    fig = plt.figure(figsize=(3,6))
    eval = [i+1 for i in range(len(df_all))]
    for method_i in method:
        plt.plot(df_all["num_eval"], df_all[method_i+"_q3"],color=c[method_i],label=method_i)
        plt.fill_between(df_all["num_eval"], df_all[method_i+"_q1"], df_all[method_i+"_q5"],color=c[method_i],alpha=0.2)
        plt.fill_between(df_all["num_eval"], df_all[method_i+"_q2"], df_all[method_i+"_q4"],color=c[method_i],alpha=0.3)
    plt.legend()
    fig.savefig(res_path+"/_fig_{}_q1_q5.png".format(name))

def fig_q2_q4(df_all,method,res_path,name):
    c = {"continuous_gep_RBF":"red",
        "discontinuous_gep_RBF":"blue",
        "continuous_gep_KRG":"green",
        "discontinuous_gep_KRG":"cyan",
        "gep":"yellow"}
    fig = plt.figure(figsize=(3,6))
    eval = [i+1 for i in range(len(df_all))]
    for method_i in method:
        plt.plot(df_all["num_eval"], df_all[method_i+"_q3"],color=c[method_i],label=method_i)
        plt.fill_between(df_all["num_eval"], df_all[method_i+"_q2"], df_all[method_i+"_q4"],color=c[method_i],alpha=0.3)
    plt.legend()
    fig.savefig(res_path+"/_fig_{}_q2_q4.png".format(name))