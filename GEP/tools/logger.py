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
    """
    LOCAL REFINEMENT
    """
    def _log_local_data(self,gen=None,eval=None,EI=None, start_time=0,f_sur=None,x_=None):
        log_list = {"generation":gen,"num_eval":eval,"time":time.time()-start_time,
                "f_sur":f_sur,"f_EI":EI}
        for dim in range(len(x_)):
            log_list["x{}".format(dim)] = x_[dim]
        return log_list

    def _log_local_refinement_save(self,df):
        df_n = pd.DataFrame(df)        # new
        if os.path.isfile(self.cnf.c_respath + "/_log_local_refinement.csv"):
            df_o = pd.read_csv(self.cnf.c_respath + "/_log_local_refinement.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.c_respath + "/_log_local_refinement.csv",index = False)
        else:
            df_n.to_csv(self.cnf.c_respath + "/_log_local_refinement.csv",index = False)
    """
    BEST LOCAL REFINEMENT(NO TIME EDITION)
    """
    #def _log_best_local_data(self,local_i=None,gen=None,eval=None,EI=None,f_sur=None,x_=None):
    #    if self.cnf.sur_mode == "RBF":
    #        idx = np.argmin(f_sur)
    #    elif self.cnf.sur_mode == "KRG":
    #        idx = np.argmax(EI)
    #    dec_var = x_[idx]
    #    log_list = {"local_i":local_i,"generation":gen,"num_eval":eval,"f_sur":f_sur[idx],"f_EI":EI[idx]}
    #    for dim in range(len(dec_var)):
    #        log_list["x{}".format(dim)] = dec_var[dim]
    #    return log_list

    def _log_best_local_data(self,local_i=None,gen=None,eval=None,EI=None,f_sur=None,x_=None):
        log_list = {"local_i":local_i,"generation":gen,"num_eval":eval,"f_sur":f_sur,"f_EI":EI}
        for dim in range(len(x_)):
            log_list["x{}".format(dim)] = x_[dim]
        return log_list

    def _log_local_refinement_best_save(self,df):
        df_n = pd.DataFrame(df)        # new
        if os.path.isfile(self.cnf.c_respath + "/_log_local_refinement_best.csv"):
            df_o = pd.read_csv(self.cnf.c_respath + "/_log_local_refinement_best.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.c_respath + "/_log_local_refinement_best.csv",index = False)
        else:
            df_n.to_csv(self.cnf.c_respath + "/_log_local_refinement_best.csv",index = False)
    """
    SURROGATE
    """
    def _log_data_surrogate(self,gen=None,eval=None,kendalltau=None,rank_dif=None,rmse=None,):
        log_list = {"generation":gen,"num_eval":eval,"kendalltau":kendalltau,"rank_dif":rank_dif, "rmse":rmse}
        return log_list
    def _log_surrogate_save(self,df):
        df_n = pd.DataFrame(df)        # new
        if os.path.isfile(self.cnf.c_respath + "/_log_surrogate.csv"):
            df_o = pd.read_csv(self.cnf.c_respath + "/_log_surrogate.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.c_respath + "/_log_surrogate.csv",index = False)
        else:
            df_n.to_csv(self.cnf.c_respath + "/_log_surrogate.csv",index = False)

    """
    ARCHIVE
    """
    def _log_archive_save(self,df):
        df_n = pd.DataFrame(df)        # new
        if os.path.isfile(self.cnf.c_respath + "/_log_archive.csv"):
            df_o = pd.read_csv(self.cnf.c_respath + "/_log_archive.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.c_respath + "/_log_archive.csv",index = False)
        else:
            df_n.to_csv(self.cnf.c_respath + "/_log_archive.csv",index = False)

    """
    FORMAT
    """
    def _log_save(self):
        df_n = pd.DataFrame(self.df_cgp_log)        # new
        self.df_cgp_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_cgp.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_cgp.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_cgp.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_cgp.csv",index = False)