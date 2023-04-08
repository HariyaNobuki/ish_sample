import argparse
import os

"""
py -3.7 main.py --lognumber 0 --numtrial 10 --mode gep --encode discontinuous --surrogate RBF
"""

def set_parse():
    parser = argparse.ArgumentParser(description='NKTLAB DISCONTINUOUS SAEA PJ')
    # nas path "//192.168.11.8/Experiment//2023_hariya"
    parser.add_argument('--NAS', '-naspath', type=str, default="local")
    parser.add_argument('--NAS_HRY', '-nas_hry', type=str, default="//192.168.11.8/Experiment//2023_hariya")

    # method 
    parser.add_argument('--mode', '-m', type=str, default='gep', help='gep , SAEA_gep')
    # log
    parser.add_argument('--lognumber', '-sm', type=int, default=0, help='log file name')
    parser.add_argument('--numtrial', '-t', type=int, default=10, help='Num. of trials')
    # problem
    parser.add_argument('--num_x', '-num_x', type=int, default=1, help='num of x')
    parser.add_argument('--x_min', '-x_min', type=int, default=-1, help='range x')
    parser.add_argument('--x_max', '-x_max', type=int, default=1, help='range x')
    parser.add_argument('--train_plot', '-train_p', type=int, default=10, help='train plot')
    parser.add_argument('--test_plot', '-test_p', type=int, default=100, help='test plot')

    # encode dependent G2V
    parser.add_argument('--vectorsize', '-vs', type=int, default=10, help='vector size')
    parser.add_argument('--maxeval', '-me', type=int, default=10000, help='maxevaluation')
    parser.add_argument('--header', '-head', type=int, default=6, help='header length')
    parser.add_argument('--n_genes', '-n_genes', type=int, default=3, help='n_genes length')
    parser.add_argument('--numpop', '-pop', type=int, default=100, help='Num. of population')

    args = parser.parse_args()
    return args