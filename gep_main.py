# Prod. By  Hariya Nobuki
# gene expression programming

## module
import crayons  
import warnings

## my module
from setting.configuration import Configuration
from setting.arg_setting import set_parse
from GEP.algorithm.gep import GEP
from GEP.tools.make_pro_plot import Problems  # plz use this file

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    args = set_parse()
    cnf = Configuration(args)
    params = cnf.set_problem()
    pro = Problems()
    for info in params:
        print(crayons.red("### "),crayons.red(info['name']))
        for trial in range(args.num_trial):
            cnf.resetSeed(trial)
            alg = GEP(cnf)
            pop = alg._init_gep()  # make plot
            pro.get_init_plot(info['name'],info['x_range'][0],info['x_range'][1],info['num_x'])
            alg.main(pop)




