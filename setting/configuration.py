import numpy as np
import random 
import os
import shutil


## problem setting
from GEP.problem import F0

class Configuration:
    def __init__(self,args):
        self.c_path = os.getcwd()
        self.args = args

        self.maxeval = args.maxeval     # df = 1500
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
            'x_range':[-1,1],
            'operand':["+", "-", "sin", "cos"],
          })
        #if "F2" in problems:
        #    params.append(F2)
        #if "F3" in problems:
        #    params.append(F3)
        #if "F5" in problems:
        #    params.append(F5)
        #if "F6" in problems:
        #    params.append(F6)
        #if "F9" in problems:
        #    params.append(F9)
        return params
    
    def set_param(self,info):
        self.operand = info['operand']
        self.train_plot = 100
        self.test_plot = 100
    
    def set_init_sample(self,mode):
        ### benchmark ###
        if mode == "train":
            X,Y = self.init_XY()
        elif mode == "test":
            X,Y = self.test_XY()
        return X,Y

    def resetSeed(self,seed=1):
        np.random.seed(seed)
        random.seed(seed)
