import torch
import time
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from src.plot.TBoard import TBoard

markers = ['o', 'v', 'd']

class CustomPlot(TBoard):

    def __init__(self, tb_activated, n_instances, plot_offset=10):
        super().__init__(tb_activated, n_instances)
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

    def update_plot_state2d_action_values(self, plot, ax, x, y, z):
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
    
        if step >= 0: # step % self._plot_offset == 0:
            self.update_plot_state2d_action_values(surf, ax, X, Y, Z)
            plt.pause(0.5)

    def plot_q_values(self, disc_states, best_actions, state_action_values, action_dict_tags, curr_action):
        x = disc_states[:, 0].numpy()
        y = disc_states[:, 1].numpy()
        N = int(len(state_action_values)**.5)
        z = state_action_values.view(N, N).numpy()
        print(z)
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap='plasma')
        ax.set_aspect('auto')
        cbar = fig.colorbar(cax)

        for i_action in range(action_dict_tags.size):
            mask = best_actions == i_action
            x_act = x[torch.nonzero(mask)]
            y_act = y[torch.nonzero(mask)]
            ax.scatter(x_act, y_act, label='Action: {}'.format(i_action), marker=markers[i_action], s=10**2)

        ax.grid(False)
        ax.set_title(curr_action)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

    def plot_state_actions(self, mc_disc, select_action, policy_net, action_dict_tags):
        disc_state_space = mc_disc.get_bins()
        disc_states = torch.cartesian_prod(torch.tensor(disc_state_space[0]), torch.tensor(disc_state_space[1]))
        best_actions = torch.tensor(list(map(select_action, disc_states)))
        print(best_actions)
        values = policy_net(disc_states)
        for i_action in range(action_dict_tags.size):
            actions = torch.full((np.prod(mc_disc.size), 1), i_action, dtype=int)
            state_action_values = values.gather(1, actions).detach()
            self.plot_q_values(disc_states, best_actions, state_action_values, action_dict_tags, action_dict_tags[i_action])
        plt.show()