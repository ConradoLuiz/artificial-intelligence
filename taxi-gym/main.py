import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os 
from scipy.signal import savgol_filter
from qAgentTabular import QLearningAgentTabular
from qAgentLinear import QLearningAgentLinear
import pickle

taxi_env = gym.make("Taxi-v3").env

linear_agent = QLearningAgentLinear(taxi_env.observation_space.n, taxi_env.action_space.n, taxi_env)

def train_agent(agent, env, total_episodes = 60_000):
  rewards = []

  for episode in range(total_episodes):
      state = env.reset()
      episode_rewards = []

      if episode % (total_episodes/20) == 0:
        print(episode)

      while True:
              
          action = agent.choose_action()
          
          # transição
          new_state, reward, done, info = env.step(action)
          
          agent.learn(state, action, reward, new_state, done, episode)

          state = new_state

          # env.render()
          
          episode_rewards.append(reward)
          
          if done == True:
            break
            
      rewards.append(np.mean(episode_rewards))
  
  return agent, rewards


linear_agent, rewards = train_agent(linear_agent, taxi_env, 100)

print(rewards)
input()

# pickle.dump( linear_agent, open( "linear_agent.p", "wb" ) )
# pickle.dump( rewards, open( "rewards.p", "wb" ) )

# plt.plot(savgol_filter(rewards, 3, 2))
# plt.show()

