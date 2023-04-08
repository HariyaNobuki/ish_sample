"""
for analysis 
PLOT
&
TEST PLOT
Manual for random number matching
"""

import numpy as np
import pandas as pd
import glob
import crayons
import os , sys
import seaborn as sns
import matplotlib.pyplot as plt

method_list = ['GEP','EQL']

dict_pl = {
'Nguyen5':'bench',
'Nguyen7':'bench',
'Keijzer11':'bench',
'Nonic':'bench',
'Hartman':'bench',
'wine':'real',
'concrete':'real',
'airfoil':'real',
'boston':'real',
}
res_path = "_result"
out_path = "_result_summary"
out_path2 = "_result_visual"

if __name__ == '__main__':
    print(crayons.red("### LOG ANALYTICS ###"))
    df_iris = sns.load_dataset("iris")
    columns_i = ["length","width","length2","width2","out"]
    df_iris.columns = columns_i

    #print(df_iris.head())
    #sns.boxplot(data=df_iris)
    #plt.show()

    os.makedirs(out_path,exist_ok=True)
    for met in method_list:
        os.makedirs(out_path+'/'+met,exist_ok=True)
    os.makedirs(out_path2,exist_ok=True)

    ### SUMMARY METHODS ###
    for problem in dict_pl.keys():
        print(crayons.blue('###'),crayons.blue(problem))
        for method in ['GEP']:
            res_list = []
            met_path = res_path + '/' + method + '/' + problem
            folderfile = os.listdir(met_path)
            for folderfile_i in folderfile:
                if glob.glob(met_path + "/"+folderfile_i+"/*.csv"):
                    csv = glob.glob(met_path + "/"+folderfile_i+"/*.csv")[-1]
                    df_i = pd.read_csv(csv)
                    try:
                        res_list.append(min(df_i['fitness']))
                    except:
                        res_list.append(min(df_i['error_list']))
        sum_loss = pd.DataFrame({'mse_loss':res_list})
        sum_loss.to_csv(out_path+'/'+method+'/'+problem+'.csv',index = False)

        for method in ['EQL']:
            res_list = []
            met_path = res_path + '/' + method + '/' + problem
            folderfile = glob.glob(met_path+'/trial*.csv')
            for folderfile_i in folderfile:
                csv = folderfile_i
                df_i = pd.read_csv(csv)
                try:
                    res_list.append(min(df_i['fitness']))
                except:
                    res_list.append(min(df_i['error_list']))
        sum_loss = pd.DataFrame({'mse_loss':res_list})
        sum_loss.to_csv(out_path+'/'+method+'/'+problem+'.csv',index = False)
    
    ### data summarize part 2
    for problem in dict_pl.keys():
        df_sum2 = pd.DataFrame()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for method in method_list:
            df = pd.read_csv(out_path+'/'+method+'/'+problem+'.csv')
            df_sum2[method] = df['mse_loss']
        # nan delete
        df_sum2 = df_sum2.dropna(how='any')
        ax.boxplot([df_sum2['EQL'], df_sum2['GEP']], labels=['EQL','GEP'])
        fig.savefig(out_path2+'/'+problem+'.pdf')
        fig.savefig(out_path2+'/'+problem+'.png')
        df_sum2.to_csv(out_path2+'/'+problem+'.csv')


    #sns.boxplot(data=df_iris)
    #plt.show()
