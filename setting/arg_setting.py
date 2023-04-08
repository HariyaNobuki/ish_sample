import argparse
import os

"""
py -3.7 main.py --lognumber 0 --numtrial 10 --mode gep --encode discontinuous --surrogate RBF
"""
## setting
def set_parse():
    parser = argparse.ArgumentParser(description='NKTLAB DISCONTINUOUS SAEA PJ')

    # logging
    parser.add_argument('--result_path', type=str, default="result/")
    # expriment setting
    parser.add_argument('--num_trial', type=int, default=31, help='Num. of trials')
    # problem
    parser.add_argument('--problems', type=list, default=["F0","F2","F3","F5","F6","F9"], help='problems')
    # GEP 
    parser.add_argument('--maxeval', '-me', type=int, default=10000, help='maxevaluation')
    parser.add_argument('--header', '-head', type=int, default=6, help='header length')
    parser.add_argument('--n_genes', '-n_genes', type=int, default=3, help='n_genes length')
    parser.add_argument('--numpop', '-pop', type=int, default=20, help='Num. of population')

    args = parser.parse_args()
    return args