
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os , sys

def saveCSV(df_METHOD,logpath,name):
    print("### CSV")
    df_METHOD.to_csv('{}/{}.csv'.format(logpath,name))

def saveFIGq1q5(df_METHOD):
    print("### FIG")
    colordict = {'gep_rbf':'red','gep_randf':'cyan','gep_rbf_nkt':'green','gep_randf_nkt':'orange','gep':'blue'}
    fig1 = plt.figure()
    fig1, ax = plt.subplots(figsize=(6, 4))
    eval = [i+1 for i in range(len(df_METHOD))]
    for me in METHOD:
        plt.plot(eval,df_METHOD["q3_"+me],lw=0.5,color=colordict[me],label=me,alpha=1)
        ax.fill_between(eval,df_METHOD["q1_"+me],df_METHOD["q5_"+me],
                        color=colordict[me],alpha=0.1)
        ax.fill_between(eval,df_METHOD["q2_"+me],df_METHOD["q4_"+me],
                        color=colordict[me],alpha=0.2)
    plt.xlabel('eval')
    plt.ylabel('fitness')
    plt.legend()
    fig1.savefig('_log/quantile_q1q5.png')

def saveFIGq2q4(df_METHOD):
    print("### FIG")
    colordict = {'gep_rbf':'red','gep_randf':'cyan','gep_randf_nkt':'orange','gep_rbf_nkt':'green','gep':'blue'}
    fig1 = plt.figure()
    fig1, ax = plt.subplots(figsize=(6, 4))
    eval = [i+1 for i in range(len(df_METHOD))]
    for me in METHOD:
        plt.plot(eval,df_METHOD["q3_"+me],lw=0.5,color=colordict[me],label=me,alpha=1)
        ax.fill_between(eval,df_METHOD["q2_"+me],df_METHOD["q4_"+me],
                        color=colordict[me],alpha=0.2)
    plt.xlabel('eval')
    plt.ylabel('fitness')
    plt.legend()
    fig1.savefig('_log/quantile_q2q4.png')

def saveFIGKEN(df_all,log_path):
    print("### FIG")
    fig1 = plt.figure()
    fig1, ax = plt.subplots(figsize=(6, 4))
    eval = [i+1 for i in range(len(df_all))]
    for tri in range(num_trial):
        plt.plot(eval,df_all["kendalltau_"+str(tri)],marker='x',ms=2.0,lw=0.5,color='red',alpha=1)
    plt.xlabel('eval')
    plt.ylabel('kendalltau')
    plt.ylim(-1,1)
    plt.legend()
    fig1.savefig('{}/kendalltau.png'.format(log_path))

def saveeach(df,num):
    fig1 = plt.figure()
    plt.plot(df['eval'],df['fitness'])
    plt.xlabel('eval')
    plt.ylabel('fitness')
    plt.legend()
    fig1.savefig('quantile_{}.png'.format(num))
    a=0

if __name__ == '__main__':
    #matplotlib.use('Agg')
    num_trial = 11
    os.makedirs('_log',exist_ok=True)

    # main analysis file -> "_analysis"
    df_METHOD = pd.DataFrame()  # save stack
    METHOD = ['gep_rbf','gep_rbf_nkt','gep_randf','gep_randf_nkt','gep']
    #METHOD = ['gep_rbf','gep_rbf_nkt','gep_randf']
    for me in METHOD:
        df_all = pd.DataFrame()
        # GEP-RBf
        log_path = '_log_' + me
        for tri in range(num_trial):
            df = pd.read_csv(log_path + '/trial_' + str(tri) + '/ev2fit.csv')
            df_all['fitness_{}'.format(tri)] = df['fitness']
        q1 = df_all.quantile(0,axis=1)
        q2 = df_all.quantile(0.25,axis=1)
        q3 = df_all.quantile(0.5,axis=1)
        q4 = df_all.quantile(0.75,axis=1)
        q5 = df_all.quantile(1,axis=1)
        df_METHOD['q1_{}'.format(me)] = q1
        df_METHOD['q2_{}'.format(me)] = q2
        df_METHOD['q3_{}'.format(me)] = q3
        df_METHOD['q4_{}'.format(me)] = q4
        df_METHOD['q5_{}'.format(me)] = q5
    saveCSV(df_METHOD,log_path,"quantile")
    saveFIGq1q5(df_METHOD)
    saveFIGq2q4(df_METHOD)

    # kendalltau
    METHOD = ['gep_rbf','gep_rbf_nkt','gep_randf','gep_randf_nkt']
    for me in METHOD:
        df_all = pd.DataFrame()
        # GEP-RBf
        log_path = '_log_' + me
        for tri in range(num_trial):
            df = pd.read_csv(log_path + '/trial_' + str(tri) + '/gen2ken.csv')
            df_all['kendalltau_{}'.format(tri)] = df['kendalltau']
        saveCSV(df_all,log_path,"kendalltau")
        saveFIGKEN(df_all,log_path)

    print("###FINISH###")