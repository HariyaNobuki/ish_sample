# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This :mod:`basic` module provides fundamental boilerplate GEP algorithm implementations. After registering proper
operations into a :class:`deap.base.Toolbox` object, the GEP evolution can be simply launched using the present
algorithms. Of course, for complicated problems, you may want to define your own algorithms, and the implementation here
can be used as a reference.
"""
import deap_hry as deap
import random
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import distance
from pyDOE2 import lhs
from scipy.spatial import distance
from scipy.stats import rankdata
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor  # randf

import pickle   # If different data types can be stored



def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def _apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


def gep_rbf(population,population_val, toolbox,MAX_EVAL, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    G_pop = 200
    next_elite_archive = 2
    next_unc_archive = 1
    localRainForce = 150
    #encode_max = (population[0].head_length + population[0].tail_length)*2 # full_encode
    encode_max = (population[0].head_length)*2
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals','num_arc'] + (stats.fields if stats else [])

    allone = [1 for i in range(encode_max)]
    alltwo = [2 for i in range(encode_max)] 
    allthree = [3 for i in range(encode_max)]
    allfour = [4 for i in range(encode_max)]
    allfive = [5 for i in range(encode_max)]

    fit_val = []
    kendalltau_val = []
    kendalltau_dict = pd.DataFrame()
    # Val Actual Evaluation
    invalid_validations = [ind for ind in population_val if not ind.fitness.valid]  # validation pop の操作をしている
    fitnesses = toolbox.map(toolbox.evaluate, invalid_validations)
    for ind, fit in zip(invalid_validations, fitnesses):
        ind.fitness.values = fit
        fit_val.append(fit[-1])
    for ind in population_val:
        ind_enc = []
        for gene in range(len(ind)):
            for i in range(ind.head_length):
                ind_enc.append(ind[gene].head[i]._encode_name)
            ind.RBF_encode = np.array(ind_enc)
    # encode x_val
    x_val = np.array([])
    for ind in population_val:
        if x_val.shape == (0,):
            x_val = np.expand_dims(ind.RBF_encode,0)
        else:
            x_val = np.append(x_val, np.array([ind.RBF_encode]),axis=0)
    fit_rank_val = rankdata(fit_val)

    ARCHIVE_X = np.array([])
    ARCHIVE_F = np.array([])
    ARCHIVE_POP = []
    ARCHIVE_DIV = []
    fit_list = []
    fit_dict = pd.DataFrame()

    for gen in range(n_generations + 1):
        # init generation
        if gen == 0:
            # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]  # この式を触るとvalid変数を操作することになる
            fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
                fit_list.append(fit[-1])
                fit_list[-1] = min(fit_list)
            #ARCHIVE_POP = population
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)

            for ind in population:
                if ARCHIVE_X.shape == (0,):
                    ARCHIVE_X = np.expand_dims(ind.RBF_encode,0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
                else:
                    if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                        print("BUT")
                        continue
                    elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                        print("ALL")
                        continue
                    else:
                        ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                        ARCHIVE_POP.append(ind)
                        ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                        ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
            aVar = ARCHIVE_X
            aObj = ARCHIVE_F
            data = np.vstack((aVar.T, aObj.T))
            rbf = Rbf(*data, function='cubic')

            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  rbf(*(x_.T))
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]

            population = deap.tools.selBest(population, k=G_pop)    # screening
        # record statistics and log
        if hall_of_fame is not None:        # default is TRUE
            hall_of_fame.update(population)

        # make logbook(dinamic log)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals),num_arc=len(ARCHIVE_POP),
                        **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break

        for G in range(localRainForce):
            # selection with elitism
            elites = deap.tools.selRBFBest(population, k=n_elites)
            offspring = deap.tools.selRBFTournament(population, len(population) - n_elites)

            # replication
            offspring = [toolbox.clone(ind) for ind in offspring]

            # mutation
            for op in toolbox.pbs:
                if op.startswith('mut'):
                    offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # crossover
            for op in toolbox.pbs:
                if op.startswith('cx'):
                    offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # replace the current population with the offsprings
            population = elites + offspring

            # update RBF Fitness
            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  rbf(*(x_.T))
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]
                
        ### after surrogate ###
        # rbf_uncertainly
        goto_archive = []

        dist_unc_list = []
        for ind in population:
            dist_unc = distance.cdist(ARCHIVE_X, [ind.RBF_encode], metric='euclidean')
            dist_unc_list.append(max(dist_unc[0,:]))
        for order in range(next_unc_archive):
            rbf_uncertainly = population[np.argmax(dist_unc_list)]
            goto_archive.append(rbf_uncertainly)
            dist_unc_list.pop(np.argmax(dist_unc_list))

        # rbf_elite
        rbf_elites = deap.tools.selRBFBest(population, k=next_elite_archive) # rbf_best individual 
        for ind in range(len(rbf_elites)):
            goto_archive.append(rbf_elites[ind])

        # Evaluate the object
        fitnesses = toolbox.map(toolbox.evaluate, goto_archive)
        for ind, fit in zip(goto_archive, fitnesses):
            ind.fitness.values = fit
            fit_list.append(fit[-1])
            fit_list[-1] = min(fit_list)

        # finishing ??
        if len(fit_list) >= MAX_EVAL:
            break

        # go to archive
        for ind in goto_archive:
            ind_enc = []
            for gene in range(len(ind)):
                for i in range(ind.head_length):
                    ind_enc.append(ind[gene].head[i]._encode_name)
                #for i in range(ind.tail_length):
                #    ind_enc.append(ind[gene].tail[i]._encode_name)
                ind.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("BUT")
                continue
            else:   # add ARCHIVE
                if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                    print("BUT")
                    continue
                elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                    print("ALL")
                    continue
                else:
                    ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
        aVar = ARCHIVE_X
        aObj = ARCHIVE_F
        data = np.vstack((aVar.T, aObj.T))
        rbf = Rbf(*data, function='cubic')

        # validation Kendour tau
        rbf_fitness_val =  rbf(*(x_val.T))
        rbf_rank_val = rankdata(rbf_fitness_val)
        correlation, pvalue = kendalltau(fit_rank_val,rbf_rank_val)
        kendalltau_val.append(correlation)

        population = deap.tools.selBest(ARCHIVE_POP, k=G_pop)
    
    ARCHIVE_DICT = pd.DataFrame(ARCHIVE_DIV,columns=['q1','q2','q3','q4','q5'])
    ARCHIVE_DICT['eval'] = [i for i in range(len(ARCHIVE_DICT))]
    fit_dict['fitness'] = fit_list
    fit_dict['eval'] = [i for i in range(len(fit_list))]
    kendalltau_dict['kendalltau'] = kendalltau_val
    kendalltau_dict['gen'] = [i for i in range(len(kendalltau_val))]
    return population,logbook,ARCHIVE_F,ARCHIVE_DICT,fit_dict,kendalltau_dict\


def gep_randf(population,population_val, toolbox,MAX_EVAL, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    forest = RandomForestRegressor(n_estimators=100,
                                #criterion='mse', 
                                max_depth=None, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                bootstrap=True, 
                                oob_score=False, 
                                n_jobs=None, 
                                random_state=None, 
                                verbose=0, 
                                warm_start=False, 
                                ccp_alpha=0.0, 
                                max_samples=None
                                )

    G_pop = 200
    next_elite_archive = 2
    next_unc_archive = 1
    localRainForce = 150

    encode_max = (population[0].head_length)*2
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals','num_arc'] + (stats.fields if stats else [])

    allone = [1 for i in range(encode_max)]
    alltwo = [2 for i in range(encode_max)] 
    allthree = [3 for i in range(encode_max)]
    allfour = [4 for i in range(encode_max)]
    allfive = [5 for i in range(encode_max)]

    fit_val = []
    kendalltau_val = []
    kendalltau_dict = pd.DataFrame()
    # Val Actual Evaluation
    invalid_validations = [ind for ind in population_val if not ind.fitness.valid]  # validation pop の操作をしている
    fitnesses = toolbox.map(toolbox.evaluate, invalid_validations)
    for ind, fit in zip(invalid_validations, fitnesses):
        ind.fitness.values = fit
        fit_val.append(fit[-1])
    for ind in population_val:
        ind_enc = []
        for gene in range(len(ind)):
            for i in range(ind.head_length):
                ind_enc.append(ind[gene].head[i]._encode_name)
            ind.RBF_encode = np.array(ind_enc)
    # encode x_val
    x_val = np.array([])
    for ind in population_val:
        if x_val.shape == (0,):
            x_val = np.expand_dims(ind.RBF_encode,0)
        else:
            x_val = np.append(x_val, np.array([ind.RBF_encode]),axis=0)
    fit_rank_val = rankdata(fit_val)

    ARCHIVE_X = np.array([])
    ARCHIVE_F = np.array([])
    ARCHIVE_POP = []
    ARCHIVE_DIV = []
    fit_list = []
    fit_dict = pd.DataFrame()

    for gen in range(n_generations + 1):
        # init generation
        if gen == 0:
            # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]  # この式を触るとvalid変数を操作することになる
            fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
                fit_list.append(fit[-1])
                fit_list[-1] = min(fit_list)
            #ARCHIVE_POP = population
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)

            for ind in population:
                if ARCHIVE_X.shape == (0,):
                    ARCHIVE_X = np.expand_dims(ind.RBF_encode,0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
                else:
                    if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                        print("BUT")
                        continue
                    elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                        print("ALL")
                        continue
                    else:
                        ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                        ARCHIVE_POP.append(ind)
                        ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                        ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
            forest.fit(ARCHIVE_X, ARCHIVE_F)
            #aVar = ARCHIVE_X
            #aObj = ARCHIVE_F
            #data = np.vstack((aVar.T, aObj.T))
            #rbf = Rbf(*data, function='cubic')

            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            #rbf_fitness =  rbf(*(x_.T))
            rbf_fitness =  forest.predict(x_)
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]

            population = deap.tools.selBest(population, k=G_pop)    # screening
        # record statistics and log
        if hall_of_fame is not None:        # default is TRUE
            hall_of_fame.update(population)

        # make logbook(dinamic log)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals),num_arc=len(ARCHIVE_POP),
                        **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break

        for G in range(localRainForce):
            # selection with elitism
            elites = deap.tools.selRBFBest(population, k=n_elites)
            offspring = deap.tools.selRBFTournament(population, len(population) - n_elites)

            # replication
            offspring = [toolbox.clone(ind) for ind in offspring]

            # mutation
            for op in toolbox.pbs:
                if op.startswith('mut'):
                    offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # crossover
            for op in toolbox.pbs:
                if op.startswith('cx'):
                    offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # replace the current population with the offsprings
            population = elites + offspring

            # update RBF Fitness
            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  forest.predict(x_)
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]
                
        ### after surrogate ###
        # rbf_uncertainly
        goto_archive = []

        dist_unc_list = []
        for ind in population:
            dist_unc = distance.cdist(ARCHIVE_X, [ind.RBF_encode], metric='euclidean')
            dist_unc_list.append(max(dist_unc[0,:]))
        for order in range(next_unc_archive):
            rbf_uncertainly = population[np.argmax(dist_unc_list)]
            goto_archive.append(rbf_uncertainly)
            dist_unc_list.pop(np.argmax(dist_unc_list))

        # rbf_elite
        rbf_elites = deap.tools.selRBFBest(population, k=next_elite_archive) # rbf_best individual 
        for ind in range(len(rbf_elites)):
            goto_archive.append(rbf_elites[ind])

        # Evaluate the object
        fitnesses = toolbox.map(toolbox.evaluate, goto_archive)
        for ind, fit in zip(goto_archive, fitnesses):
            ind.fitness.values = fit
            fit_list.append(fit[-1])
            fit_list[-1] = min(fit_list)

        # finishing ??
        if len(fit_list) >= MAX_EVAL:
            break

        # go to archive
        for ind in goto_archive:
            ind_enc = []
            for gene in range(len(ind)):
                for i in range(ind.head_length):
                    ind_enc.append(ind[gene].head[i]._encode_name)
                #for i in range(ind.tail_length):
                #    ind_enc.append(ind[gene].tail[i]._encode_name)
                ind.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("BUT")
                continue
            else:   # add ARCHIVE
                if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                    print("BUT")
                    continue
                elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                    print("ALL")
                    continue
                else:
                    ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
        forest.fit(ARCHIVE_X, ARCHIVE_F)
        #aVar = ARCHIVE_X
        #aObj = ARCHIVE_F
        #data = np.vstack((aVar.T, aObj.T))
        #rbf = Rbf(*data, function='cubic')

        # validation Kendour tau
        #rbf_fitness_val =  rbf(*(x_val.T))
        rbf_fitness_val = forest.predict(x_val)
        rbf_rank_val = rankdata(rbf_fitness_val)
        correlation, pvalue = kendalltau(fit_rank_val,rbf_rank_val)
        kendalltau_val.append(correlation)

        population = deap.tools.selBest(ARCHIVE_POP, k=G_pop)
    
    ARCHIVE_DICT = pd.DataFrame(ARCHIVE_DIV,columns=['q1','q2','q3','q4','q5'])
    ARCHIVE_DICT['eval'] = [i for i in range(len(ARCHIVE_DICT))]
    fit_dict['fitness'] = fit_list
    fit_dict['eval'] = [i for i in range(len(fit_list))]
    kendalltau_dict['kendalltau'] = kendalltau_val
    kendalltau_dict['gen'] = [i for i in range(len(kendalltau_val))]
    return population,logbook,ARCHIVE_F,ARCHIVE_DICT,fit_dict,kendalltau_dict


def gep_rbf_nkt(population,population_val, toolbox,MAX_EVAL, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    G_pop = 200
    next_elite_archive = 2
    next_unc_archive = 1
    localRainForce = 150
    encode_max = (population[0].head_length)*2
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals','num_arc'] + (stats.fields if stats else [])

    allone = [1 for i in range(encode_max)]
    alltwo = [2 for i in range(encode_max)] 
    allthree = [3 for i in range(encode_max)]
    allfour = [4 for i in range(encode_max)]
    allfive = [5 for i in range(encode_max)]

    fit_val = []
    kendalltau_val = []
    kendalltau_dict = pd.DataFrame()
    # Val Actual Evaluation
    invalid_validations = [ind for ind in population_val if not ind.fitness.valid]  # validation pop の操作をしている
    fitnesses = toolbox.map(toolbox.evaluate, invalid_validations)
    for ind, fit in zip(invalid_validations, fitnesses):
        ind.fitness.values = fit
        fit_val.append(fit[-1])
    for ind in population_val:
        ind_enc = []
        for gene in range(len(ind)):
            for i in range(ind.head_length):
                ind_enc.append(ind[gene].head[i]._encode_name)
            ind.RBF_encode = np.array(ind_enc)
    # encode x_val
    x_val = np.array([])
    for ind in population_val:
        if x_val.shape == (0,):
            x_val = np.expand_dims(ind.RBF_encode,0)
        else:
            x_val = np.append(x_val, np.array([ind.RBF_encode]),axis=0)
    fit_rank_val = rankdata(fit_val)


    ARCHIVE_X = np.array([])
    ARCHIVE_F = np.array([])
    ARCHIVE_POP = []
    ARCHIVE_DIV = []
    fit_list = []
    fit_dict = pd.DataFrame()

    for gen in range(n_generations + 1):
        # init generation
        if gen == 0:
            # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]  # この式を触るとvalid変数を操作することになる
            fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
                fit_list.append(fit[-1])
                fit_list[-1] = min(fit_list)
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)

            for ind in population:
                if ARCHIVE_X.shape == (0,):
                    ARCHIVE_X = np.expand_dims(ind.RBF_encode,0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
                else:
                    if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                        print("BUT")
                        continue
                    elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                        print("ALL")
                        continue
                    else:
                        ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                        ARCHIVE_POP.append(ind)
                        ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                        ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
            aVar = ARCHIVE_X
            aObj = ARCHIVE_F
            data = np.vstack((aVar.T, aObj.T))
            rbf = Rbf(*data, function='cubic')

            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  rbf(*(x_.T))
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]

            population = deap.tools.selBest(population, k=G_pop)
        # record statistics and log
        if hall_of_fame is not None:        # default is TRUE
            hall_of_fame.update(population)

        # make logbook(dinamic log)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals),num_arc=len(ARCHIVE_POP),
                        **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break

        for G in range(localRainForce):
            # selection with elitism
            elites = deap.tools.selRBFBest(population, k=n_elites) # とりあえず最良個体を取ってきてくれた

            offspring = deap.tools.selRBFTournament(population, len(population) - n_elites)
            #offspring = deap.tools.selRBFTournament(population,k=300)

            # replication
            offspring = [toolbox.clone(ind) for ind in offspring]

            # mutation
            for op in toolbox.pbs:
                if op.startswith('mut'):
                    offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # crossover
            for op in toolbox.pbs:
                if op.startswith('cx'):
                    offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # replace the current population with the offsprings
            population = elites + offspring

            # update RBF Fitness
            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  rbf(*(x_.T))
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]
                
        # after surrogate
        # rbf_uncertainly
        goto_archive = []

        dist_unc_list = []
        for ind in population:
            dist_unc = distance.cdist(ARCHIVE_X, [ind.RBF_encode], metric='euclidean')
            dist_unc_list.append(max(dist_unc[0,:]))
        for order in range(next_unc_archive):
            rbf_uncertainly = population[np.argmax(dist_unc_list)]
            goto_archive.append(rbf_uncertainly)
            dist_unc_list.pop(np.argmax(dist_unc_list))

        # rbf_elite (ver NKT)
        cand_pool = deap.tools.selRBFBest(population, k=len(population)) # rbf_best individual 
        while len(goto_archive) <= next_elite_archive+next_unc_archive:
            rbf_elites = cand_pool[0]
            ind_enc = []
            # encode
            for gene in range(len(rbf_elites)):
                for i in range(rbf_elites.head_length):
                    ind_enc.append(rbf_elites[gene].head[i]._encode_name)
                rbf_elites.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("There is in ARC")
                cand_pool.pop(0)
            elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                print("There is in ARC")
                cand_pool.pop(0)
            else:
                goto_archive.append(rbf_elites)


        fitnesses = toolbox.map(toolbox.evaluate, goto_archive)
        for ind, fit in zip(goto_archive, fitnesses):
            ind.fitness.values = fit
            fit_list.append(fit[-1])
            fit_list[-1] = min(fit_list)

        # finishing ??
        if len(fit_list) >= MAX_EVAL:
            break

        # go to archive
        for ind in goto_archive:
            ind_enc = []
            for gene in range(len(ind)):
                for i in range(ind.head_length):
                    ind_enc.append(ind[gene].head[i]._encode_name)
                ind.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("BUT")
                continue
            elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                print("ALL")
                continue
            else:
                ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                ARCHIVE_POP.append(ind)
                ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
        aVar = ARCHIVE_X
        aObj = ARCHIVE_F
        data = np.vstack((aVar.T, aObj.T))
        rbf = Rbf(*data, function='cubic')

        # validation Kendour tau
        rbf_fitness_val =  rbf(*(x_val.T))
        rbf_rank_val = rankdata(rbf_fitness_val)
        correlation, pvalue = kendalltau(fit_rank_val,rbf_rank_val)
        kendalltau_val.append(correlation)

        population = deap.tools.selBest(ARCHIVE_POP, k=G_pop)
    
    ARCHIVE_DICT = pd.DataFrame(ARCHIVE_DIV,columns=['q1','q2','q3','q4','q5'])
    ARCHIVE_DICT['eval'] = [i for i in range(len(ARCHIVE_DICT))]
    fit_dict['fitness'] = fit_list
    fit_dict['eval'] = [i for i in range(len(fit_list))]
    kendalltau_dict['kendalltau'] = kendalltau_val
    kendalltau_dict['gen'] = [i for i in range(len(kendalltau_val))]
    return population,logbook,ARCHIVE_F,ARCHIVE_DICT,fit_dict,kendalltau_dict


def gep_randf_nkt(population,population_val, toolbox,MAX_EVAL, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    forest = RandomForestRegressor(n_estimators=100,
                                #criterion='mse', 
                                max_depth=None, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                bootstrap=True, 
                                oob_score=False, 
                                n_jobs=None, 
                                random_state=None, 
                                verbose=0, 
                                warm_start=False, 
                                ccp_alpha=0.0, 
                                max_samples=None
                                )
    G_pop = 200
    next_elite_archive = 2
    next_unc_archive = 1
    localRainForce = 150
    encode_max = (population[0].head_length)*2
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals','num_arc'] + (stats.fields if stats else [])

    allone = [1 for i in range(encode_max)]
    alltwo = [2 for i in range(encode_max)] 
    allthree = [3 for i in range(encode_max)]
    allfour = [4 for i in range(encode_max)]
    allfive = [5 for i in range(encode_max)]

    fit_val = []
    kendalltau_val = []
    kendalltau_dict = pd.DataFrame()
    # Val Actual Evaluation
    invalid_validations = [ind for ind in population_val if not ind.fitness.valid]  # validation pop の操作をしている
    fitnesses = toolbox.map(toolbox.evaluate, invalid_validations)
    for ind, fit in zip(invalid_validations, fitnesses):
        ind.fitness.values = fit
        fit_val.append(fit[-1])
    for ind in population_val:
        ind_enc = []
        for gene in range(len(ind)):
            for i in range(ind.head_length):
                ind_enc.append(ind[gene].head[i]._encode_name)
            ind.RBF_encode = np.array(ind_enc)
    # encode x_val
    x_val = np.array([])
    for ind in population_val:
        if x_val.shape == (0,):
            x_val = np.expand_dims(ind.RBF_encode,0)
        else:
            x_val = np.append(x_val, np.array([ind.RBF_encode]),axis=0)
    fit_rank_val = rankdata(fit_val)


    ARCHIVE_X = np.array([])
    ARCHIVE_F = np.array([])
    ARCHIVE_POP = []
    ARCHIVE_DIV = []
    fit_list = []
    fit_dict = pd.DataFrame()

    for gen in range(n_generations + 1):
        # init generation
        if gen == 0:
            # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]  # この式を触るとvalid変数を操作することになる
            fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
                fit_list.append(fit[-1])
                fit_list[-1] = min(fit_list)
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)

            for ind in population:
                if ARCHIVE_X.shape == (0,):
                    ARCHIVE_X = np.expand_dims(ind.RBF_encode,0)
                    ARCHIVE_POP.append(ind)
                    ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                    ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
                else:
                    if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                        print("BUT")
                        continue
                    elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                        print("ALL")
                        continue
                    else:
                        ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                        ARCHIVE_POP.append(ind)
                        ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                        ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
            forest.fit(ARCHIVE_X, ARCHIVE_F)
            #aVar = ARCHIVE_X
            #aObj = ARCHIVE_F
            #data = np.vstack((aVar.T, aObj.T))
            #rbf = Rbf(*data, function='cubic')

            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness = forest.predict(x_)
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]

            population = deap.tools.selBest(population, k=G_pop)
        # record statistics and log
        if hall_of_fame is not None:        # default is TRUE
            hall_of_fame.update(population)

        # make logbook(dinamic log)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals),num_arc=len(ARCHIVE_POP),
                        **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break

        for G in range(localRainForce):
            # selection with elitism
            elites = deap.tools.selRBFBest(population, k=n_elites) # とりあえず最良個体を取ってきてくれた

            offspring = deap.tools.selRBFTournament(population, len(population) - n_elites)
            #offspring = deap.tools.selRBFTournament(population,k=300)

            # replication
            offspring = [toolbox.clone(ind) for ind in offspring]

            # mutation
            for op in toolbox.pbs:
                if op.startswith('mut'):
                    offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # crossover
            for op in toolbox.pbs:
                if op.startswith('cx'):
                    offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

            # replace the current population with the offsprings
            population = elites + offspring

            # update RBF Fitness
            x_ = np.array([])
            for ind in population:
                ind_enc = []
                for gene in range(len(ind)):
                    for i in range(ind.head_length):
                        ind_enc.append(ind[gene].head[i]._encode_name)
                    #for i in range(ind.tail_length):
                    #    ind_enc.append(ind[gene].tail[i]._encode_name)
                    ind.RBF_encode = np.array(ind_enc)
            for ind in population:
                if x_.shape == (0,):
                    x_ = np.expand_dims(ind.RBF_encode,0)
                else:
                    x_ = np.append(x_, np.array([ind.RBF_encode]),axis=0)
            rbf_fitness =  forest.predict(x_)
            for pop in range(len(population)):
                population[pop].rbf_fitness = rbf_fitness[pop]
                
        # after surrogate
        # rbf_uncertainly
        goto_archive = []

        dist_unc_list = []
        for ind in population:
            dist_unc = distance.cdist(ARCHIVE_X, [ind.RBF_encode], metric='euclidean')
            dist_unc_list.append(max(dist_unc[0,:]))
        for order in range(next_unc_archive):
            rbf_uncertainly = population[np.argmax(dist_unc_list)]
            goto_archive.append(rbf_uncertainly)
            dist_unc_list.pop(np.argmax(dist_unc_list))

        # rbf_elite (ver NKT)
        cand_pool = deap.tools.selRBFBest(population, k=len(population)) # rbf_best individual 
        while len(goto_archive) <= next_elite_archive+next_unc_archive:
            rbf_elites = cand_pool[0]
            ind_enc = []
            for gene in range(len(rbf_elites)):
                for i in range(rbf_elites.head_length):
                    ind_enc.append(rbf_elites[gene].head[i]._encode_name)
                #for i in range(rbf_elites.tail_length):
                #    ind_enc.append(rbf_elites[gene].tail[i]._encode_name)
                rbf_elites.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("There is in ARC")
                cand_pool.pop(0)
            elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                print("There is in ARC")
                cand_pool.pop(0)
            else:
                goto_archive.append(rbf_elites)


        fitnesses = toolbox.map(toolbox.evaluate, goto_archive)
        for ind, fit in zip(goto_archive, fitnesses):
            ind.fitness.values = fit
            fit_list.append(fit[-1])
            fit_list[-1] = min(fit_list)

        # finishing ??
        if len(fit_list) >= MAX_EVAL:
            break

        # go to archive
        for ind in goto_archive:
            ind_enc = []
            for gene in range(len(ind)):
                for i in range(ind.head_length):
                    ind_enc.append(ind[gene].head[i]._encode_name)
                #for i in range(ind.tail_length):
                #    ind_enc.append(ind[gene].tail[i]._encode_name)
                ind.RBF_encode = np.array(ind_enc)
            if (np.sum(ARCHIVE_X == ind_enc , axis=1) == encode_max).any():
                print("BUT")
                continue
            elif((ind_enc == allthree)or(ind_enc == allfour)or(ind_enc == allfive)):
                print("ALL")
                continue
            else:
                ARCHIVE_X = np.append(ARCHIVE_X, np.array([ind.RBF_encode]),axis=0)
                ARCHIVE_POP.append(ind)
                ARCHIVE_F = np.append(ARCHIVE_F , ind.fitness.values[-1])
                ARCHIVE_DIV.append([np.quantile(ARCHIVE_F,0),np.quantile(ARCHIVE_F,0.25),np.quantile(ARCHIVE_F,0.50),np.quantile(ARCHIVE_F,0.75),np.quantile(ARCHIVE_F,1,00)])
        forest.fit(ARCHIVE_X, ARCHIVE_F)
        #aVar = ARCHIVE_X
        #aObj = ARCHIVE_F
        #data = np.vstack((aVar.T, aObj.T))
        #rbf = Rbf(*data, function='cubic')

        # validation Kendour tau
        rbf_fitness_val = forest.predict(x_val)
        rbf_rank_val = rankdata(rbf_fitness_val)
        correlation, pvalue = kendalltau(fit_rank_val,rbf_rank_val)
        kendalltau_val.append(correlation)

        population = deap.tools.selBest(ARCHIVE_POP, k=G_pop)
    
    ARCHIVE_DICT = pd.DataFrame(ARCHIVE_DIV,columns=['q1','q2','q3','q4','q5'])
    ARCHIVE_DICT['eval'] = [i for i in range(len(ARCHIVE_DICT))]
    fit_dict['fitness'] = fit_list
    fit_dict['eval'] = [i for i in range(len(fit_list))]
    kendalltau_dict['kendalltau'] = kendalltau_val
    kendalltau_dict['gen'] = [i for i in range(len(kendalltau_val))]
    return population,logbook,ARCHIVE_F,ARCHIVE_DICT,fit_dict,kendalltau_dict


def gep_simple(population, toolbox,MAX_EVAL, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    fit_list = []
    fit_dict = pd.DataFrame()

    for gen in range(n_generations + 1):
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
            fit_list.append(fit[-1])
            fit_list[-1] = min(fit_list)

        # finishing ??
        if len(fit_list) >= MAX_EVAL:
            break

        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(population, k=n_elites)
        offspring = toolbox.select(population, len(population) - n_elites)

        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring

    fit_dict['fitness'] = fit_list
    fit_dict['eval'] = [i for i in range(len(fit_list))]

    return population,logbook,fit_dict

__all__ = ['_validate_basic_toolbox','_apply_modification','_apply_crossover',
            'gep_randf','gep_rbf_nkt','gep_randf_nkt','gep_simple']


