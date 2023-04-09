### module
import numpy as np

class Problems:
    def __init__(self) -> None:
        pass
    def get_init_plot(self,pro_name,x_min,x_max,num_x,train_plot=100):
        x = np.random.uniform(x_min,x_max, size=(num_x,train_plot))
        t_x = np.random.uniform(x_min - 0.1 ,x_max + 0.1, size=(num_x,train_plot))
        if pro_name == 'F0':
            y,t_y = self.F0(x)

        return x,y,t_x,t_y

    def F0(self,x,t_x):
        y = np.sin(x[0] + x[1]) - np.cos(x[0] - x[1])
        t_y = np.sin(t_x[0] + t_x[1]) - np.cos(t_x[0] - t_x[1])
        return y,t_y
