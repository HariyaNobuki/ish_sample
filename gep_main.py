# Prod. By  Hariya Nobuki
# gene expression programming

## module
import crayons  
import warnings

## my module
from setting.configuration import Configuration
from setting.arg_setting import set_parse
from GEP.algorithm.gep import GEP
from GEP.tools.makefiles import MakeFiles

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    args = set_parse()
    cnf = Configuration(args)
    params = cnf.set_problem()
    for info in params:
        print(crayons.red("### "),crayons.red(info['name']))
        for trial in range(args.num_trial):
            cnf.resetSeed(trial)
            alg = GEP(cnf)
            pop = alg._init_gep()  # make plot
            alg.main(pop)




