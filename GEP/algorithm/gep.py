import os , sys
import crayons
import operator 
from operator import attrgetter
import numpy as np
import pandas as pd
import time
import tqdm

from scipy.interpolate import Rbf
from scipy.spatial import distance
from pyDOE2 import lhs
from scipy.stats import rankdata
from scipy.stats import kendalltau , norm
from sklearn.ensemble import RandomForestRegressor  # randf
from sklearn.metrics import r2_score

import pickle   # If different data types can be stored
### My module
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.interpolate import Rbf
from    pydacefit.dace                      import DACE
import  GPy

sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import geppy_hry as gep
from deap_hry import creator, base, tools
import deap_hry as deap

import myoperation


class GEP:
    def __init__(self,cnf):
        self.cnf = cnf
        self.Logger = logger.Logger(cnf)

        self.num_eval = 0
        # for save df list
        self.df_main_log = []
    
    def main(self):
        # symbolic regression
        ### version 2 ###
        input_names = ["x{}".format(i) for i in range(self.cnf.num_x)]
        pset = gep.PrimitiveSet('Main', input_names=input_names)  # gep : symbol
        pset.add_function(operator.add, 2)
        pset.add_function(operator.sub, 2)
        pset.add_function(operator.mul, 2)
        pset.add_function(myoperation.protected_div, 2)
        pset.add_function(myoperation.sin, 1)
        pset.add_function(myoperation.cos, 1)
        pset.add_function(myoperation.exp, 1)
        #pset.add_function(myoperation.log, 1)
        #pset.add_ephemeral_terminal(name='enc', gen=lambda: np.random.uniform(0, 1)) # each ENC is a random integer within [-10, 10]
        
        creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin , sur_encode = [] , f_sur = 0. , f_EI = 0.)

        self.toolbox = gep.Toolbox()     # geppy from self.toolbox
        self.toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=self.cnf.h)
        self.toolbox.register('individual', creator.Individual, gene_gen=self.toolbox.gene_gen, n_genes=self.cnf.n_genes, linker=myoperation.addLinker)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # compile utility: which translates an individual into an executable function (Lambda)
        self.toolbox.register('compile', gep.compile_, pset=pset)
        self.toolbox.register('evaluate', self.evaluate)
        self.toolbox.register('test_evaluate', self.test_evaluate)
        self.toolbox.register('select', tools.selTournament, tournsize=5)

        # 1. general operators
        self.toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.051, pb=1)
        self.toolbox.register('mut_invert', gep.invert, pb=0.1)
        self.toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
        self.toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
        self.toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
        self.toolbox.register('cx_1p', gep.crossover_one_point, pb=0.2)
        self.toolbox.register('cx_2p', gep.crossover_two_point, pb=0.5)
        self.toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
        self.toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: expected one point mutation in an individual
        self.toolbox.pbs['mut_ephemeral'] = 1  # we can also give the probability via the pbs property

        # 2. setting
        # size of population and number of generations
        pop = self.toolbox.population(n=self.cnf.n_pop)
        hof = tools.HallOfFame(1)   # only record the best three individuals ever found in all generations

        # start evolution
        self.cnf.resetSeed()
        self.cnf.n_gen = int((self.cnf.MAX_EVAL/self.cnf.n_pop))
        pop = self.gep(pop ,
                        MAX_EVAL = self.cnf.MAX_EVAL, n_generations=self.cnf.n_gen,
                        stats=None, hall_of_fame=hof, verbose=False)

    def evaluate(self,individual):
        """ variation any number """
        # zip??
        """Evalute the fitness of an individual: MAE (mean absolute error)"""
        func = self.toolbox.compile(individual)
        x = [self.X[i] for i in range(self.cnf.num_x)]
        Yp = np.array(list(map(func, *x)))
        return np.sqrt(np.mean((self.Y - Yp)**2)/self.Y.shape[0]),

    def test_evaluate(self,individual):
        """Evalute the fitness of an individual: MAE (mean absolute error)"""
        func = self.toolbox.compile(individual)
        x = [self.test_X[i] for i in range(self.cnf.num_x)]
        Yp = np.array(list(map(func, *x)))
        return np.sqrt(np.mean((self.test_Y - Yp)**2)/self.Y.shape[0]),
    
    def MakeInitPlot(self):
        # set problem
        self.X,self.Y = self.cnf.set_init_sample("train")
        self.test_X,self.test_Y = self.cnf.set_init_sample("test")


    def gep(self,population,MAX_EVAL, n_generations,
                stats=None, hall_of_fame=None, verbose=__debug__):

        gep._validate_basic_toolbox(self.toolbox)

        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals','num_arc','fitness','k_tau'] + (stats.fields if stats else [])

        # emergency escape root
        fit_list = []

        for gen in tqdm.tqdm(range(n_generations + 1)):
            self.cnf.resetSeed()
            # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
            test_fitnesses = self.toolbox.map(self.toolbox.test_evaluate, invalid_individuals)
            for ind, fit, t_fit in zip(invalid_individuals, fitnesses, test_fitnesses):
                ind.fitness.values = fit
                fit_list.append(fit[-1])
                fit_list[-1] = min(fit_list)
                self.num_eval += 1
                self.df_main_log.append(self.Logger._log_main_data(gen=gen,eval=self.num_eval,
                                        fitness=fit_list[-1],test_fitness=t_fit[-1]))
            
            # finishing 
            if self.num_eval >= MAX_EVAL:
                self.Logger._log_main_data_save(self.df_main_log)
                break

            # selection with elitism
            elites = deap.tools.selBest(population, k=self.cnf.n_elites)
            offspring = self.toolbox.select(population, len(population) - self.cnf.n_elites)

            # replication
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # mutation
            for op in self.toolbox.pbs:
                if op.startswith('mut'):
                    offspring = gep._apply_modification(offspring, getattr(self.toolbox, op), self.toolbox.pbs[op])

            # crossover
            for op in self.toolbox.pbs:
                if op.startswith('cx'):
                    offspring = gep._apply_crossover(offspring, getattr(self.toolbox, op), self.toolbox.pbs[op])

            # replace the current population with the offsprings
            population = elites + offspring
        self.Logger._log_main_data_save(self.df_main_log)
        return population