import numpy as np
import random 
import os
import shutil

## my module
from GEP.tools.makefiles import MakeFiles

## problem setting
from GEP.problem import F0

class Configuration:
    def __init__(self,args):
        self.c_path = os.getcwd()
        self.args = args

        self.MAX_EVAL = args.maxeval     # df = 1500
        self.h = args.header          # head length
        self.n_genes = args.n_genes    # number of genes in a chromosome
        self.n_pop = args.numpop
        self.n_elites = 1

        self.res_root = self.args.result_path
    
    def set_problem(self):
        params = []
        problems = self.args.problems
        if "F0" in problems:
            params.append({
            'name':'F0',
            'num_x':2,
            'x1':[-1,1],
            'x2':[-1,1],
            'operand':["+", "-", "sin", "cos"],
          })
        #if "F2" in problems:
        #    params.append(F2._set_problem_dict)
        #if "F3" in problems:
        #    params.append(F3._set_problem_dict)
        #if "F5" in problems:
        #    params.append(F5._set_problem_dict)
        #if "F6" in problems:
        #    params.append(F6._set_problem_dict)
        #if "F9" in problems:
        #    params.append(F9._set_problem_dict)
        return params
    
    def set_param(self,info):
        self.num_x = info['num_x']
        self.operand = info['operand']



    
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
