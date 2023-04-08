import crayons
import numpy as np
import random 
import os,sys
import shutil
from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic
import warnings

# make dataset
from load_dataset import load_airfoil
from load_dataset import load_boston
from load_dataset import load_wine
from load_dataset import load_winered
from load_dataset import load_winewhite
from load_dataset import load_concrete
from load_dataset import load_yacht
import load_problem

sys.path.append(os.path.join(os.path.dirname(__file__), 'setting'))
import _edit_profile


# locak debaugs
class Configuration:
    def __init__(self,args):
        self.c_path = os.getcwd()
        self.args = args
        warnings.simplefilter('ignore')
        self.dict_pl = {
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

        if self.args.NAS == "local":
            os.makedirs("_result",exist_ok=True)
            self.ex_path = "_result"
            self.c_path = os.getcwd()

        self.c_seed = None
        #desktop_path = os.path.expanduser('~/Desktop') + "\192.168.11.8\Experiment\\2023_hariya"
        self.MAX_EVAL = self.args.maxeval     # df = 1500
        self.h = self.args.header          # head length
        self.n_genes = self.args.n_genes    # number of genes in a chromosome
        self.train_plot = self.args.train_plot
        self.test_plot = self.args.test_plot
        self.num_trial = self.args.numtrial
        self.n_pop = self.args.numpop
        self.n_pop_val = 100
        self.n_gen = 10000

        self.n_elites = 1
        self.logsplit = 10
    
    def problem_setting(self):
        if self.problem == 'Nguyen5':
            p_args = _edit_profile.Nguyen5()
        elif self.problem == 'Nguyen7':
            p_args = _edit_profile.Nguyen7()
        elif self.problem == 'Keijzer11':
            p_args = _edit_profile.Keijzer11()
        elif self.problem == 'Nonic':
            p_args = _edit_profile.Nonic()
        elif self.problem == 'Hartman':
            p_args = _edit_profile.Hartman()
        elif self.problem == 'wine':
            p_args = _edit_profile.Keijzer11()
        else:
            p_args = _edit_profile.DAMMY_REAL()
        self.train_plot = p_args.N_TRAIN
        self.test_plot = p_args.N_TEST
        self.num_x = p_args.X_DIM
        self.args.x_min = p_args.X_MIN
        self.args.x_max = p_args.X_MAX

    
    def ex_reset(self,path):
        try:
            shutil.rmtree(path+"/")
        except OSError as e:
            pass
    
    def set_c_respath(self,respath):
        self.c_respath = respath
    
    def set_problem(self,problem):
        self.problem = problem
        self.res_path = self.ex_path + "/GEP/" + self.problem

    
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

    def resetSeed(self,seed=10000):
        seed = self.c_seed
        np.random.seed(seed=seed)
        random.seed(seed)
