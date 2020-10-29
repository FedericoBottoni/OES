import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from TBoard import TBoard

class CustomPlot(TBoard):

    def __init__(self, tb_activated, plot_offset=10):
        super().__init__(tb_activated)
        # if tb_activated:
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        self._state2d_action_values_plots = list()
        self._plot_offset = plot_offset
    
    def add_state2d_action_values_plots(self, tags):
        for tag in tags:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(tag)
            X, Y, Z = (np.array([0, 1]), np.array([0, 1]), np.zeros([2, 2]))
            surf = [ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')]
            fig.colorbar(surf[0], shrink=0.5, aspect=5)
            plt.draw()
            self._state2d_action_values_plots.append(list((tag, fig, ax, surf, X, Y, Z)))

    def update_plot(self, plot, ax, x, y, z):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, z, cmap="magma")

    def push_state2d_action_values(self, tag, data_tpl, step):
        elem_index = -1
        for elem_i in range(len(self._state2d_action_values_plots)):
            if self._state2d_action_values_plots[elem_i][0] == tag:
                elem_index = elem_i
        tag, _, ax, surf, X, Y, Z = self._state2d_action_values_plots[elem_index]
        
        Z[data_tpl[0], data_tpl[1]] = data_tpl[2]
        
        self._state2d_action_values_plots[elem_index][4] = X
        self._state2d_action_values_plots[elem_index][5] = Y
        self._state2d_action_values_plots[elem_index][6] = Z
        
        print("x", X)
        print("y", Y)
        print("z", Z)
        if step >= 0: # step % self._plot_offset == 0:
            self.update_plot(surf, ax, X, Y, Z)
            plt.pause(0.5)