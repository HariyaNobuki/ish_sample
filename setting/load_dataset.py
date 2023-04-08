import pandas as pd
import numpy as np

def load_airfoil(path):
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    dec_names = 'Frequency','Angle','Length','Velocity','Thickness'
    df_data = pd.read_table(path + '/airfoil_self_noise.dat',
                        names=('Frequency','Angle','Length','Velocity','Thickness','Sound'))
    X_ = np.array([])
    for d_name in dec_names:
        if X_.shape == (0,):
            X_ = np.expand_dims(df_data['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([df_data['{}'.format(d_name)]]), axis=0)
    return X_ , np.array(df_data['Sound'])

def load_boston(path):
    from sklearn.datasets import load_boston
    data = load_boston()
    X = data["data"]
    y = data["target"]
    feature_names = data["feature_names"]
    boston_df = pd.DataFrame(data=X, columns=feature_names)
    X_ = np.array([])
    for d_name in boston_df:
        if X_.shape == (0,):
            X_ = np.expand_dims(boston_df['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([boston_df['{}'.format(d_name)]]), axis=0)
    return X_,y

def load_wine(path):
    # https://atmarkit.itmedia.co.jp/ait/articles/2208/25/news046.html
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data["data"]
    y = data["target"]
    feature_names = data["feature_names"]
    wine_df = pd.DataFrame(data=X, columns=feature_names)
    X_ = np.array([])
    for d_name in wine_df:
        if X_.shape == (0,):
            X_ = np.expand_dims(wine_df['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([wine_df['{}'.format(d_name)]]), axis=0)
    return X_,y

def load_winered(path):
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    dec_names = 'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'
    df_data = pd.read_csv(path + '/winequality-red.csv')
    X_ = np.array([])
    for d_name in dec_names:
        if X_.shape == (0,):
            X_ = np.expand_dims(df_data['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([df_data['{}'.format(d_name)]]), axis=0)
    return X_ , np.array(df_data['quality'])

def load_winewhite(path):
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    dec_names = 'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'
    df_data = pd.read_csv(path + '/winequality-white.csv')
    X_ = np.array([])
    for d_name in dec_names:
        if X_.shape == (0,):
            X_ = np.expand_dims(df_data['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([df_data['{}'.format(d_name)]]), axis=0)
    return X_ , np.array(df_data['quality'])

def load_concrete(path):
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    dec_names = 'Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age'
    df_data = pd.read_csv(path + '/concrete_data.csv')
    X_ = np.array([])
    for d_name in dec_names:
        if X_.shape == (0,):
            X_ = np.expand_dims(df_data['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([df_data['{}'.format(d_name)]]), axis=0)
    return X_ , np.array(df_data['Strength'])

def load_yacht(path):
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    dec_names = 'Position','Coefficient','Length-displacement','Length-beam','Froude'
    df_data = pd.read_csv(path + '/yacht_hydrodynamics.csv',
                        names=('Position','Coefficient','Length-displacement','Length-beam','Froude','Residuary'))
    X_ = np.array([])
    for d_name in dec_names:
        if X_.shape == (0,):
            X_ = np.expand_dims(df_data['{}'.format(d_name)],0)
        else:
            X_ = np.append(X_,np.array([df_data['{}'.format(d_name)]]), axis=0)
    return X_ , np.array(df_data['Residuary'])
