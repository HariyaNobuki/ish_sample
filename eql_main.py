"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import _edit_profile
import numpy as np
import os , sys

# utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'DSR'))
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNet, MaskedSymbolicNet
from utils.regularization import l12_smooth
# setting
sys.path.append(os.path.join(os.path.dirname(__file__), 'setting'))
import load_problem
import load_dataset

import time
import argparse
import pandas as pd

import warnings
warnings.simplefilter('ignore')


# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.1
init_sd_last = 1.0
init_sd_middle = 0.5

class Benchmark:
    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            *[functions.Cos()] * 2,
            *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]

        self.n_layers = n_layers              # Number of hidden layers
        self.reg_weight = reg_weight     # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000    # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir


    def benchmark(self, func_name, trials):

        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        # Create a new sub-directory just for the specific function
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Train network!
        expr_list, time_list = self.train(func_name, trials, func_dir)
        df = pd.DataFrame({
            'expr' : expr_list,
            'time' : time_list,
        })
        df.to_csv(os.path.join(func_dir, '_{}.csv'.format(func_name)))


    def train(self, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""
        # Setting up the symbolic regression network
        x_dim = p_args.X_DIM  # Number of input arguments to the function
        x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
        sym = SymbolicNet(self.n_layers,
                          funcs=self.activation_funcs,
                          initial_weights=[tf.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
                                           tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.truncated_normal([width, 1], stddev=init_sd_last)])
        # sym = SymbolicNet(self.n_layers, funcs=self.activation_funcs)
        y_hat = sym(x_placeholder)

        # Label and errors
        error = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
        error_test = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat)
        reg_loss = l12_smooth(sym.get_weights())
        #loss = error + self.reg_weight * reg_loss
        loss = error        # main regression

        # Set up TensorFlow graph for training
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train = opt.minimize(loss)


        eq_list = []
        time_list = []

        # Only take GPU memory as needed - allows multiple jobs on a single GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for trial in range(trials):
                # Arrays to keep track of various quantities as a function of epoch
                loss_list = []  # Total loss (MSE + regularization)
                error_list = []     # MSE
                reg_list = []       # Regularization
                error_test_list = []    # Test error

                print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(trials))

                loss_val = np.nan
                # Restart training if loss goes to NaN (which happens when gradients blow up)
                while np.isnan(loss_val):
                    sess.run(tf.global_variables_initializer())

                    t0 = time.time()
                    # First stage of training, preceded by 0th warmup stage
                    for i in range(self.n_epochs1 + 2000):
                        if i < 2000:
                            #lr_i = self.learning_rate * 10
                            lr_i = self.learning_rate 
                        else:
                            lr_i = self.learning_rate

                        feed_dict = {x_placeholder: x, learning_rate: lr_i}
                        _ = sess.run(train, feed_dict=feed_dict)
                        if i % self.summary_step == 0:
                            loss_val, error_val, reg_val, = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            reg_list.append(reg_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                # Arrays to keep track of various quantities as a function of epoch
                                loss_list = []  # Total loss (MSE + regularization)
                                error_list = []     # MSE
                                reg_list = []       # Regularization
                                error_test_list = []    # Test error
                                time_list = []
                                break

                    t1 = time.time()

                    # Masked network - weights below a threshold are set to 0 and frozen. This is the fine-tuning stage
                    sym_masked = MaskedSymbolicNet(sess, sym)
                    y_hat_masked = sym_masked(x_placeholder)
                    error_masked = tf.losses.mean_squared_error(labels=y, predictions=y_hat_masked)
                    error_test_masked = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat_masked)
                    train_masked = opt.minimize(error_masked)

                    # 2nd stage of training
                    t2 = time.time()
                    for i in range(self.n_epochs2):
                        feed_dict = {x_placeholder: x, learning_rate: self.learning_rate / 10}
                        _ = sess.run(train_masked, feed_dict=feed_dict)
                        if i % self.summary_step == 0:
                            loss_val, error_val = sess.run((loss, error_masked), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test_masked, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break
                    t3 = time.time()
                tot_time = t1-t0 + t3-t2
                #print(tot_time)
                time_list.append(tot_time)

                # Print the expressions
                weights = sess.run(sym_masked.get_weights())
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])

                # Save results
                trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)
                df_results = pd.DataFrame({
                    #"weights": weights,
                    "loss_list": loss_list,
                    "error_list": error_list,
                    "error_test": error_test_list,
                })
                df_results.to_csv(os.path.join(func_dir, 'trial%d.csv' % trial))

                eq_list.append(expr)

        return eq_list, time_list

def init_XY():
    X = np.random.uniform(p_args.X_MIN,p_args.X_MAX, size=(p_args.X_DIM,p_args.N_TRAIN))   # random numbers in range [-10, 10)
    Y = Load_Problem.switcher(X)
    return X , Y

if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='_result/EQL')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=10001, help="Number of epochs to train the first stage")
    parser.add_argument("--n-epochs2", type=int, default=10001,
                        help="Number of epochs to train the second stage, after freezing weights.")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')


    dict_pl = {
                'Nguyen5':'bench',
                'Nguyen7':'bench',
                'Keijzer11':'bench',
                'Nonic':'bench',
                'Hartman':'bench',
                'wine':'real',
                'concrete':'real',
                'airfoil':'real',
                'boston':'real'
                }

    for problem in dict_pl.keys():

        if problem == 'Nguyen5':
            p_args = _edit_profile.Nguyen5()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        elif problem == 'Nguyen7':
            p_args = _edit_profile.Nguyen7()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        elif problem == 'Keijzer11':
            p_args = _edit_profile.Keijzer11()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        elif problem == 'Nonic':
            p_args = _edit_profile.Nonic()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        elif problem == 'Hartman':
            p_args = _edit_profile.Hartman()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        elif problem == 'wine':
            p_args = _edit_profile.Keijzer11()
            var_names = ["x_{}".format(i) for i in range(p_args.X_DIM)]
        else:
            p_args = _edit_profile.DAMMY_REAL()

        if dict_pl[problem] == 'bench':
            Load_Problem = load_problem.Problem(problem)
            x , y = init_XY()
            x_test, y_test = init_XY()
        elif dict_pl[problem] == 'real':
            if problem == 'wine':
                x , y = load_dataset.load_wine(None)
                x_test, y_test = load_dataset.load_wine(None)
                var_names = ["x_{}".format(i) for i in range(x.shape[0])]
                p_args.X_DIM = x.shape[0]
            elif problem == 'concrete':
                x , y = load_dataset.load_concrete(os.getcwd()+"/dataset/")
                x_test, y_test = load_dataset.load_concrete(os.getcwd()+"/dataset/")
                var_names = ["x_{}".format(i) for i in range(x.shape[0])]
                p_args.X_DIM = x.shape[0]
            elif problem == 'boston':
                x , y = load_dataset.load_boston(None)
                x_test, y_test = load_dataset.load_boston(None)
                var_names = ["x_{}".format(i) for i in range(x.shape[0])]
                p_args.X_DIM = x.shape[0]
            elif problem == 'airfoil':
                x , y = load_dataset.load_airfoil(None)
                x_test, y_test = load_dataset.load_airfoil(None)
                var_names = ["x_{}".format(i) for i in range(x.shape[0])]
                p_args.X_DIM = x.shape[0]
        x , x_test = x.T , x_test.T
        for i in range(x.shape[1]):
            x[:,i] = x[:,i]/x[:,i].max()
            x_test[:,i] = x_test[:,i]/x_test[:,i].max()
        y , y_test= y.reshape([y.shape[0], 1]),y_test.reshape([y_test.shape[0], 1]),

        bench = Benchmark(**kwargs)
        bench.benchmark(func_name=problem, trials=10)





