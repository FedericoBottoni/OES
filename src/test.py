import json
import gym
import numpy as np
import torch

from src.DQN import DQN

def test():
    with open('config.json') as json_file:
        config = json.load(json_file)
        gym_environment = config['gym_environment']
        save_model_path = config['save_model']['path']

    env = gym.make(gym_environment)
    observation = env.reset()
    obs_length = observation.shape[0]
    n_actions = env.action_space.n

    model = DQN(obs_length, n_actions)
    model.load_state_dict(torch.load(save_model_path))
    model.eval()
    
    def select_action(state):
        with torch.no_grad():
            return model(state).max(1)[1]

    for _ in range(20):
        currentState = env.reset().reshape(1, 2)
        rewardSum=0
        for t in range(300):
            env.render()
            action = select_action(torch.from_numpy(currentState).float())

            new_state, reward, done, _ = env.step(action.item())

            new_state = new_state.reshape(1, 2)

            currentState=new_state

            rewardSum+=reward
            if done:
                print("Episode finished after {} timesteps reward is {}".format(t+1,rewardSum))
                break
    env.close()