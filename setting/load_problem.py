import pandas as pd
import numpy as np

class Problem:
    def __init__(self,problem) -> None:
        self.problem = problem
    
    def switcher(self,X):
        if self.problem == 'Nguyen5':
            return self.Nguyen5(X)
        elif self.problem == 'Nguyen7':
            return self.Nguyen7(X)
        elif self.problem == 'Keijzer11':
            return self.Keijzer11(X)
        elif self.problem == 'Keijzer14':
            return self.Keijzer14(X)
        elif self.problem == 'Nonic':
            return self.Nonic(X)
        elif self.problem == 'Hartman':
            return self.Hartman(X)
        elif self.problem == 'Koza2':
            return self.Koza2(X)
        else:
            print("ERROR")

    def Koza2(self,x):
        return x[0]**5 - 2*(x[0]**3) + x[0]

    def Nguyen5(self,x):
        return np.sin(x[0]**2) * np.cos(x[0])-1

    def Nguyen7(self,x):
        return np.log(x[0] + 1) + np.log(x[0]**2 + 1)

    def Keijzer11(self,x):
        return x[0]*x[1] + np.sin((x[0]-1)-(x[1]-1))

    def Keijzer14(self,x):
        return 8/(2+(x[0]**2)+(x[1]**2))

    def Nonic(self,x):
        return x[0] + x[0]**2 + x[0]**3 + x[0]**4 + x[0]**5 + x[0]**6 + x[0]**7 + x[0]**8 + x[0]**9

    def Hartman(self,x):
        return -np.exp(-((x[0]**2)+(x[1]**2)+(x[2]**2)+(x[3]**2)))