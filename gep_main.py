# all algorithm for gen fitness
# this variation 2 (x , y)
import os , sys
import setting.configuration
import setting.arg_setting


from GEP.algorithm import gep
from GEP.tools.makefiles import MakeFiles

import numpy as np
import crayons
import _edit_profile

if __name__ == '__main__':

    args = arg_setting.set_parse()

    cnf = configuration.Configuration(args)
    cnf.resetSeed()
    
    #cnf.ex_reset(cnf.res_path)

    print(crayons.red(r"### normal gep ###"))

    for problem in cnf.dict_pl.keys():
        print(crayons.blue("### "),end='')
        print(problem)
        cnf.set_problem(problem)
        cnf.problem_setting()

        for trial in range(cnf.num_trial):
            c_trial = args.lognumber + trial
            cnf.c_seed = c_trial
            cnf.resetSeed()
            MakeFiles(filename="trial_{}".format(c_trial),path=cnf.res_path)
            cnf.set_c_respath(respath=cnf.res_path+"/trial_{}".format(c_trial))
            alg = gep.GEP(cnf)
            alg.MakeInitPlot()  # make plot
            alg.main()




