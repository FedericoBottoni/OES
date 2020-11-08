import gym
import math
import random
import numpy as np
import time
import atexit
import json
from functools import partial
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F

import src.early_stopping as early_stopping
from src.plot.CustomPlot import CustomPlot
from src.ReplayMemory import ReplayMemory
from src.DQN import DQN
from src.transfer.PTL import PTL
from src.preprocessing.MountainCarDiscretizer import MountainCarDiscretizer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def run():
    start = time.time()
    with open('config.json') as json_file:
        config = json.load(json_file)
        enable_plots = config['enable_plots']
        gym_environment = config['gym_environment']
        action_dict = config['action_names']
        action_dict_tags = np.array(list(action_dict.items()))[:, 1]
        num_episodes = config['training_episodes']
        max_steps = config['max_steps']
        stop_condition = [config['stop_condition']['reward_threshold'], config['stop_condition']['n_episodes']]
        save_model = config['save_model']['active']
        transfer_hyperparams = config['transfer_hyperparams']
        hyperparams = config['network_hyperparams']
        if save_model:
            save_model_path = config['save_model']['path']

    eval_stop_condition_bound = partial(early_stopping.eval_stop_condition, stop_condition)

    n_instances = 1
    if transfer_hyperparams != None:
        n_instances = transfer_hyperparams['N_PROCESSES']

    env = [None] * n_instances
    observation = [None] * n_instances
    policy_net = [None] * n_instances
    target_net = [None] * n_instances
    optimizer = [None] * n_instances
    memory = [None] * n_instances
    state = [None] * n_instances
    
    ALPHA = hyperparams['ALPHA']
    EPS_END = hyperparams['EPS_END']
    EPS_DECAY = hyperparams['EPS_DECAY']
    EPS_START = hyperparams['EPS_START']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    GAMMA = hyperparams['GAMMA']
    
    for i in range(n_instances):
        if max_steps != None and max_steps > 0:
            env[i] = gym.make(gym_environment)
            env[i]._max_episode_steps = max_steps
        else:
            env[i] = gym.make(gym_environment).unwrapped
        observation[i] = env[i].reset()

    c_plot = CustomPlot(enable_plots)
    mc_disc = MountainCarDiscretizer(env[0], [3, 3])
    ptl = None
    if transfer_hyperparams:
        ptl = PTL(mc_disc, n_instances, transfer_hyperparams)
    
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_length = observation[0].shape[0]
    n_actions = env[0].action_space.n
    
    for p in range(n_instances):
        policy_net[p] = DQN(obs_length, n_actions).to(device)
        target_net[p] = DQN(obs_length, n_actions).to(device)
        target_net[p].load_state_dict(policy_net[p].state_dict())
        target_net[p].eval()
        
        optimizer[p] = optim.Adam(policy_net[p].parameters(), lr=ALPHA, )
        memory[p] = ReplayMemory(10000)

    def select_action(p, state, steps_done, apply_eps=True):
        sample = random.random()
        if apply_eps:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            # print(eps_threshold)
        if not apply_eps or sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net[p](state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def optimize_model(p, c_plot):
        if len(memory[p]) < BATCH_SIZE:
            return None
        transitions = memory[p].sample(BATCH_SIZE)
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
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
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
    i_earlystop = 0
    
    i_episode = np.zeros([n_instances], dtype=np.int16)
    cm_reward = np.zeros([n_instances])
    ep_step = np.zeros([n_instances], dtype=np.int16)
    loss = np.zeros([n_instances])
    ep_cm_reward_dict = {}
    last_ep_i_step = 0
    procs_done = np.zeros([n_instances])
    
    atexit.register(dispose, c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode)
    for i_step in count():
        for p in range(n_instances):
            if i_episode[p] == num_episodes:
                procs_done[p] = 1
                pass
            
            if ep_step[p] == 0:
                observation[p] = env[p].reset()
                state[p] = torch.from_numpy(observation[p]).float()


            action = select_action(p, state[p], i_step)
            observation[p], reward, done, _ = env[p].step(action.item())

            reward = torch.tensor([reward], device=device)
            cm_reward[p] += reward

            if not done:
                next_state = torch.from_numpy(observation[p]).float()
            else:
                next_state = None

            # Store the transition in memory
            memory[p].push(state[p], action, next_state, reward)

            if ptl:
                ptl.update_state_visits(p, state[p])

            # Move to the next state
            state[p] = next_state

            # Perform one step of the optimization (on the target network)
            loss[p] = optimize_model(p, c_plot)
            if not math.isnan(loss[0]) and p == n_instances - 1:
                c_plot.push_loss(loss.mean())
                loss = np.zeros([n_instances])

            if done:
                print('Env #', p, 'has solved the episode', i_episode[p])
                ep_cm_reward_dict, last_cm_rewards = sync_cm_rewards(p, c_plot, ep_cm_reward_dict, i_episode, cm_reward, \
                    n_instances, i_step - last_ep_i_step)
                if last_cm_rewards.size != 0:
                    early_stop, i_earlystop = eval_stop_condition_bound(last_cm_rewards.mean(), i_earlystop)
                last_ep_i_step = i_step
                cm_reward[p] = 0
                ep_step[p] = 0
                i_episode[p] += 1
            else:
                ep_step[p] += 1

            if i_episode[p] % hyperparams['TARGET_UPDATE'] == 0:
                target_net[p].load_state_dict(policy_net[p].state_dict())
            
        if not ptl is None:
            ptl.transfer(policy_net, i_step)

        if len(np.nonzero(procs_done)[0]) == n_instances:
            break

        c_plot.push_cm_reward(cm_reward.mean())

        if early_stop:
            early_stopping.on_stop(i_episode.mean())
            break

    end = time.time()
    print('Time elapsed', int(end - start), 's')
    best_env = 0
    select_action_bound = lambda st : select_action(best_env, st, 0, apply_eps=False).item()
    c_plot.plot_state_actions(mc_disc, select_action_bound, policy_net[best_env], action_dict_tags)
    
    dispose(c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode)


def sync_cm_rewards(p, c_plot, ep_cm_reward_dict, i_episode, cm_reward, n_instances, i_step):
    ep_key = str(i_episode[p])
    if not ep_key in ep_cm_reward_dict:
        ep_cm_reward_dict[ep_key] = [None] * n_instances
    
    ep_cm_reward_dict[ep_key][p] = cm_reward[p]

    if not None in ep_cm_reward_dict[ep_key]:
        removed = np.array(ep_cm_reward_dict[ep_key])
        print('Closing episode', ep_key, 'with cm_rew', removed)
        c_plot.push_cm_reward_ep(removed.mean())
        c_plot.push_episode_len(i_step+1)
        ep_cm_reward_dict.pop(ep_key)
    else:
        removed = np.array([])
        
    return ep_cm_reward_dict, removed

def dispose(c_plot, save_model, save_model_path, policy_net, n_instances, env, start, i_episode):
    if save_model:
        env_saved = np.argmax(i_episode)
        torch.save(policy_net[env_saved].state_dict(), save_model_path)
        print('Model #', env_saved, 'saved in:', save_model_path)
    for p in range(n_instances):
        #env[p].render()
        env[p].close()
        print('Closing env #', p, 'at episode', i_episode[p])
    c_plot.dispose()   