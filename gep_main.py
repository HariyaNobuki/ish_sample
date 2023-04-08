# Prod. By  Hariya Nobuki
# gene expression programming

## module
import crayons  

## my module
from setting.configuration import Configuration
from setting.arg_setting import set_parse
from GEP.algorithm import gep
from GEP.tools.makefiles import MakeFiles

if __name__ == '__main__':

    args = set_parse()
    cnf = Configuration(args)
    cnf.resetSeed()

    for problem in cnf.dict_pl.keys():
        print(crayons.blue("### "),crayons.red(problem))
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




