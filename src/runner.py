import gym
import math
import random
import numpy as np
import json
from functools import partial
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F

import early_stopping
from TBoard import TBoard
from ReplayMemory import ReplayMemory
from DQN import DQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def run():

    with open('config.json') as json_file:
        config = json.load(json_file)
        enable_plots = config['enable_plots']
        gym_environment = config['gym_environment']
        action_dict = config['action_names']
        action_dict_tags = np.array(list(action_dict.items()))[:, 1]
        num_episodes = config['training_episodes']
        stop_condition = [config['stop_condition']['reward_threshold'], config['stop_condition']['n_episodes']]
        save_model = config['save_model']['active']
        hyperparams = config['network_hyperparams']
        if save_model:
            save_model_path = config['save_model']['path']

    eval_stop_condition_bound = partial(early_stopping.eval_stop_condition, stop_condition)

    env = gym.make(gym_environment).unwrapped

    c_plot = TBoard(enable_plots)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observation = env.reset()

    obs_length = observation.shape[0]
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(obs_length, n_actions).to(device)
    target_net = DQN(obs_length, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # c_plot.add_state2d_action_values_plots(action_dict_tags)
    # for tag in action_dict_tags:
    #    print(tag)
    # for i in range(10):
    #     c_plot.push_state2d_action_values(action_dict_tags[0], (i % 2, i % 3, i), i)
    # print("done")

    ALPHA = hyperparams['ALPHA']

    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA, )
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state, steps_done):
        sample = random.random()
        EPS_END = hyperparams['EPS_END']
        EPS_DECAY = hyperparams['EPS_DECAY']
        EPS_START = hyperparams['EPS_START']
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        # print(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def optimize_model(c_plot):
        BATCH_SIZE = hyperparams['BATCH_SIZE']
        GAMMA = hyperparams['GAMMA']
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch.view(-1, obs_length)).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states.view(-1, obs_length)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Plot
        c_plot.push_q_values(action_dict, state_action_values, action_batch)
        c_plot.push_loss(loss.item())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    early_stop = False
    i_earlystop = 0
    for i_episode in range(num_episodes):

        # Initialize the environment and state
        cm_reward = torch.zeros([1])
        observation = env.reset()
        state = torch.from_numpy(observation).float()
        for i_step in count():

            # Select and perform an action + observe new state
            action = select_action(state, steps_done)
            steps_done += 1

            observation, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            cm_reward += reward
            if not done:
                next_state = torch.from_numpy(observation).float()
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(c_plot)

            c_plot.push_cm_reward(cm_reward.item())
            if done:
                early_stop, i_earlystop = eval_stop_condition_bound(cm_reward, i_earlystop)
                c_plot.push_cm_reward_ep(cm_reward.item())
                c_plot.push_episode_len(i_step+1)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % hyperparams['TARGET_UPDATE'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if early_stop:
            early_stopping.on_stop(i_episode)
            break
    if save_model:
        torch.save(policy_net.state_dict(), save_model_path)
        print('Model saved in:', save_model_path)
    print('Complete')
    env.render()
    env.close()