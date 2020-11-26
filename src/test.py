import json
import gym
import numpy as np
import torch

from src.DQN import DQN

def test(argv):
    with open('config.json') as json_file:
        config = json.load(json_file)
        gym_environment = config['gym_environment']
        max_steps = config['max_steps']
        save_model_path = config['save_model']['path']
    n = ''
    if len(argv) > 1:
        n = argv[1]
    path = save_model_path + n + '.pth'
    print('Running', path)
    env = gym.make(gym_environment)
    observation = env.reset()
    obs_length = observation.shape[0]
    n_actions = env.action_space.n

    model = DQN(obs_length, n_actions)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    def select_action(state):
        with torch.no_grad():
            return model(state).max(1)[1]

    for _ in range(20):
        currentState = env.reset().reshape(1, obs_length)
        rewardSum=0
        i = 0
        for t in range(max_steps):
            env.render()
            action = select_action(torch.from_numpy(currentState).float())

            new_state, reward, done, _ = env.step(action.item())

            new_state = new_state.reshape(1, obs_length)

            currentState=new_state

            rewardSum+=reward
            if done:
                i = t
                break
        print("Episode finished after {} timesteps reward is {}".format(i+1,rewardSum))
    env.close()