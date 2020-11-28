import gym
import math
import random
import numpy as np
import time
import atexit
import json
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.EarlyStopping import EarlyStopping
from src.plot.CustomPlot import CustomPlot
from src.ReplayMemory import ReplayMemory
from src.DQN import DQN
from src.preprocessing.CartPoleDiscretizer import CartPoleDiscretizer
from src.transfer.visits_filters import PTL

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def run():
    start = time.time()
    with open('config.json') as json_file:
        config = json.load(json_file)
        enable_plots = config['enable_plots']
        gym_environment = config['gym_environment']
        num_episodes = config['training_episodes']
        max_steps = config['max_steps']
        ES_REWARD = config['stop_condition']['reward_threshold']
        ES_RANGE = config['stop_condition']['n_episodes']
        save_model = config['save_model']['active']
        enable_transfer = config['enable_transfer']
        transfer_hyperparams = config['transfer_hyperparams']
        hyperparams = config['network_hyperparams']
        if save_model:
            save_model_path = config['save_model']['path']

    n_instances = transfer_hyperparams['N_PROCESSES']
    TRANSFER_APEX = transfer_hyperparams['TRANSFER_APEX']

    env = [None] * n_instances
    observation = [None] * n_instances
    policy_net = [None] * n_instances
    target_net = [None] * n_instances
    optimizer = [None] * n_instances
    replay_memory = [None] * n_instances
    transfer_memory = [None] * n_instances
    state = [None] * n_instances
    
    ALPHA = hyperparams['ALPHA']
    EPS_END = hyperparams['EPS_END']
    EPS_DECAY = hyperparams['EPS_DECAY']
    EPS_START = hyperparams['EPS_START']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    GAMMA = hyperparams['GAMMA']
    STATE_DIM_BINS = config['STATE_DIM_BINS']

    TRANSFER_INTERVAL = transfer_hyperparams['TRANSFER_INTERVAL']
    TRANSFER_DISC = transfer_hyperparams['TRANSFER_DISC']
    TRANSFER_APEX = transfer_hyperparams['TRANSFER_APEX']
    THETA_MAX = transfer_hyperparams['THETA_MAX']
    THETA_MIN = transfer_hyperparams['THETA_MIN']
    
    for i in range(n_instances):
        env[i] = gym.make(gym_environment)
        observation[i] = env[i].reset()

    env_disc = CartPoleDiscretizer(env[0], [TRANSFER_DISC] * len(env[0].get_state()))
    ptl = PTL(enable_transfer, n_instances, gym_environment, env_disc, transfer_hyperparams)
    c_plot = CustomPlot(enable_plots, ptl, n_instances)
    es = EarlyStopping(n_instances, ES_REWARD, ES_RANGE)
    
    print('Running', n_instances, 'processes')
    if(enable_transfer):
        print('Transfer enabled, THETA between', THETA_MIN, '-', THETA_MAX, 'with APEX on ep.', TRANSFER_APEX, \
            'every', TRANSFER_INTERVAL, 'steps')

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_length = observation[0].shape[0]
    n_actions = env[0].action_space.n
    action_dict = env[0].get_action_labels()
    action_dict_tags = np.array(list(action_dict.items()))[:, 1]
    
    for p in range(n_instances):
        policy_net[p] = DQN(obs_length, n_actions).to(device)
        target_net[p] = DQN(obs_length, n_actions).to(device)
        target_net[p].load_state_dict(policy_net[p].state_dict())
        target_net[p].eval()
        
        optimizer[p] = optim.Adam(policy_net[p].parameters(), lr=ALPHA, )
        replay_memory[p] = ReplayMemory(10000)
        transfer_memory[p] = ReplayMemory(32)

    def get_epsilon(x):
        return EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * x / EPS_DECAY)

    def select_action(p, state, steps_done, apply_eps=True):
        sample = random.random()
        if apply_eps:
            eps_threshold = get_epsilon(steps_done)
            # print(eps_threshold)
        if not apply_eps or sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net[p](state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def optimize_model(p, c_plot, episode):
        transfer_batch_size = round(BATCH_SIZE * ptl.get_theta(episode[p]))
        if len(replay_memory[p]) < BATCH_SIZE - transfer_batch_size:
            return None
        transitions = replay_memory[p].sample(BATCH_SIZE - transfer_batch_size)
        if len(transfer_memory[p]) >= transfer_batch_size:
            transitions_trans = ptl.gather_transfer(p, transfer_memory[p], transfer_batch_size)
            transitions.extend(transitions_trans)
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

        # Compute Q(s_t, a)
        state_action_values = policy_net[p](state_batch.view(-1, obs_length)).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(len(transitions), device=device)
        next_state_values[non_final_mask] = target_net[p](non_final_next_states.view(-1, obs_length)).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Plot
        # c_plot.push_q_values(action_dict, state_action_values, action_batch)

        # Optimize the model
        optimizer[p].zero_grad()
        loss.backward()

        for param in policy_net[p].parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer[p].step()

        return loss


    early_stop = False
    
    i_episode = np.zeros([n_instances], dtype=np.int16)
    cm_reward = np.zeros([n_instances])
    ep_step = np.zeros([n_instances], dtype=np.int16)
    loss = np.zeros([n_instances])
    ep_cm_reward_dict = {}
    procs_done = np.zeros([n_instances])
    
    atexit.register(dispose, c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode)
    for i_step in count():
        for p in range(n_instances):
            if procs_done[p] == 1:
                continue
            if i_episode[p] == num_episodes:
                procs_done[p] = 1
            
            if ep_step[p] == 0:
                observation[p] = env[p].reset()
                state[p] = torch.from_numpy(observation[p]).float()


            action = select_action(p, state[p], i_step)
            observation[p], reward, done, _ = env[p].step(action.item())

            reward = torch.tensor([reward], device=device)
            cm_reward[p] += reward

            if ep_step[p] >= max_steps:
                done = True

            if not done:
                next_state = torch.from_numpy(observation[p]).float()
            else:
                next_state = None

            # Store the transition in memory
            replay_memory[p].push(state[p], action, next_state, reward)

            if enable_transfer:
                ptl.update_state_visits(p, state[p])

            # Move to the next state
            state[p] = next_state

            # Perform one step of the optimization (on the target network)
            loss[p] = optimize_model(p, c_plot, i_episode)

            if done:
                print('Env#', p, 'has solved ep#', i_episode[p])
                ep_cm_reward_dict = sync_cm_rewards(p, c_plot, ep_cm_reward_dict, i_episode, procs_done, \
                     cm_reward, n_instances, ep_step)
                early_stop = es.eval_stop_condition(p, cm_reward[p])
                cm_reward[p] = 0
                ep_step[p] = 0
                i_episode[p] += 1
            else:
                ep_step[p] += 1

            if i_episode[p] % hyperparams['TARGET_UPDATE'] == 0:
                target_net[p].load_state_dict(policy_net[p].state_dict())
                
            if p == 0 and i_step % 2000 == 0:
                print('Epsilon', get_epsilon(i_step), 'at step', i_step)
            
            if early_stop:
                es.on_stop(p)
                procs_done[p] = 1
                early_stop = False
        
        c_plot.add_step()

        if not math.isnan(loss[0]):
            c_plot.push_ar_loss(procs_done, loss)
            loss = np.zeros([n_instances])

        # Parallel Transfer Learning updates the memories
        if enable_transfer and i_step >= TRANSFER_APEX and i_step % TRANSFER_INTERVAL == 0:
            p_transitions = ptl.provide_transfer(replay_memory, policy_net)
            for p_sender in ptl.get_senders():
                for tr in p_transitions[p_sender]:
                    transfer_memory[ptl.get_receiver(p_sender)].push_t(tr)

        c_plot.push_ar_cm_reward(procs_done, cm_reward)

        if len(np.nonzero(procs_done)[0]) == n_instances:
            break

    end = time.time()
    print('Time elapsed', int(end - start), 's')
    if obs_length == 2:
        best_env = 0
        select_action_bound = lambda st : select_action(best_env, st, 0, apply_eps=False).item()
        c_plot.plot_state_actions(env_disc, select_action_bound, policy_net[best_env], action_dict_tags)
    
    dispose(c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode)


def sync_cm_rewards(p, c_plot, ep_cm_reward_dict, i_episode, procs_done, cm_reward, n_instances, i_step):
    ep_key = str(i_episode[p])
    if not ep_key in ep_cm_reward_dict:
        ep_cm_reward_dict[ep_key] = [[None, None]] * n_instances
    
    ep_cm_reward_dict[ep_key][p] = [cm_reward[p], i_step[p]]

    if not None in [ep_cm_reward_dict[ep_key][i_rew][0] for i_rew in range(n_instances) if procs_done[i_rew] == 0]:
        removed = np.array(ep_cm_reward_dict[ep_key])
        cm_rws = [i[0] for i in removed]
        lens = [i[1] for i in removed]
        print('Closing episode', ep_key, 'with cm_rew', cm_rws)
        c_plot.push_ar_cm_reward_ep(cm_rws)
        c_plot.push_ar_episode_len(lens)
        c_plot.add_episode()
        ep_cm_reward_dict.pop(ep_key)
    return ep_cm_reward_dict

def dispose(c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode):
    if save_model:
        for p in range(n_instances):
            path = save_model_path + str(p) + '.pth'
            torch.save(policy_net[p].state_dict(), path)
            print('Model #', p, 'saved in:', path)
            env[p].close()
            print('Closing env #', p, 'at episode', i_episode[p])
    c_plot.dispose()
