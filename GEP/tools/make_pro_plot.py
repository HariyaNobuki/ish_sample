### module
import numpy as np

class Problems:
    def __init__(self) -> None:
        pass
    def get_init_plot(pro_name,x_min,x_max,num_x,train_plot=100):
        x = np.random.uniform(x_min,x_max, size=(num_x,train_plot))
        if pro_name == 'F0':
            a=0

        return x , y

    def F0(self,x):
        y = 0
        return y
