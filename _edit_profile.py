import argparse

def Nguyen5():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='20')
    parser.add_argument("--N_TEST", type=int, default='1000')
    parser.add_argument("--N_VAL", type=int, default='1000')
    parser.add_argument("--X_DIM", type=int, default=1)
    parser.add_argument("--X_MIN", type=int, default=-1)
    parser.add_argument("--X_MAX", type=int, default=1)
    parser.add_argument("--NOISE_SD", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args

def Nguyen7():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='20')
    parser.add_argument("--N_TEST", type=int, default='1000')
    parser.add_argument("--N_VAL", type=int, default='1000')
    parser.add_argument("--X_DIM", type=int, default='1')
    parser.add_argument("--X_MIN", type=int, default=0)
    parser.add_argument("--X_MAX", type=int, default=2)
    parser.add_argument("--NOISE_SD", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args

def Keijzer11():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='100')
    parser.add_argument("--N_TEST", type=int, default='900')
    parser.add_argument("--N_VAL", type=int, default='900')
    parser.add_argument("--X_DIM", type=int, default='2')
    parser.add_argument("--X_MIN", type=int, default=-1)
    parser.add_argument("--X_MAX", type=int, default=1)
    parser.add_argument("--NOISE_SD", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args

def Nonic():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='20')
    parser.add_argument("--N_TEST", type=int, default='1000')
    parser.add_argument("--N_VAL", type=int, default='1000')
    parser.add_argument("--X_DIM", type=int, default='1')
    parser.add_argument("--X_MIN", type=int, default=-2)
    parser.add_argument("--X_MAX", type=int, default=2)
    parser.add_argument("--NOISE_SD", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args

def Hartman():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='100')
    parser.add_argument("--N_TEST", type=int, default='900')
    parser.add_argument("--N_VAL", type=int, default='900')
    parser.add_argument("--X_DIM", type=int, default='4')
    parser.add_argument("--X_MIN", type=int, default=0)
    parser.add_argument("--X_MAX", type=int, default=2)
    parser.add_argument("--NOISE_SD", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args

def DAMMY_REAL():
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--N_TRAIN", type=int, default='100')
    parser.add_argument("--N_TEST", type=int, default='900')
    parser.add_argument("--N_VAL", type=int, default='900')
    parser.add_argument("--X_DIM", type=int, default='4')
    parser.add_argument("--X_MIN", type=int, default=0, help="Number of hidden layers, L")
    parser.add_argument("--X_MAX", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--NOISE_SD", type=int, default=0, help="Number of hidden layers, L")
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    return args