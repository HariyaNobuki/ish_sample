import numpy as np
import random 
import os
import shutil

# setting problem
import load_problem

## my module
from GEP.tools.makefiles import MakeFiles

class Configuration:
    def __init__(self,args):
        self.c_path = os.getcwd()
        self.args = args

        self.MAX_EVAL = args.maxeval     # df = 1500
        self.h = args.header          # head length
        self.n_genes = args.n_genes    # number of genes in a chromosome
        self.train_plot = args.train_plot
        self.test_plot = args.test_plot
        self.num_trial = args.numtrial
        self.n_pop = args.numpop
        self.n_elites = 1

        self.res_root = self.args.result_path
    
    def set_problem(self,problem):
        self.problem = problem
        self.res_path = os.path.join(self.res_root , self.problem)
        MakeFiles(filename=self.res_path,path=self.res_root)

    
    def set_init_sample(self,mode):
        # data loader
        self.pl = load_problem.Problem(self.problem)
        ### real world ###
        print(os.getcwd())
        if self.problem == "airfoil":
            X,Y = load_airfoil(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "boston":
            X,Y = load_boston(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "wine":
            X,Y = load_wine(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "winered":
            X,Y = load_winered(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "winewhite":
            X,Y = load_winewhite(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "concrete":
            X,Y = load_concrete(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        elif self.problem == "yacht":
            X,Y = load_yacht(self.c_path+"/dataset/")
            self.num_x = X.shape[0]
        ### benchmark ###
        else: # ('bench')
            if mode == "train":
                X,Y = self.init_XY()
            elif mode == "test":
                X,Y = self.test_XY()
        return X,Y


    def init_XY(self):
        # np.zeros((2, NG, NT))
        X = np.random.uniform(self.args.x_min,self.args.x_max, size=(self.num_x,self.train_plot))   # random numbers in range [-10, 10)
        Y = self.pl.switcher(X)
        return X , Y

    def test_XY(self):
        X = np.random.uniform(self.args.x_min,self.args.x_max, size=(self.num_x,self.test_plot))   # random numbers in range [-10, 10)
        Y = self.pl.switcher(X)
        return X , Y

    def resetSeed(self,seed=1):
        np.random.seed(seed)
        random.seed(seed)
