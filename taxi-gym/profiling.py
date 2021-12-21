import cProfile
import gym
from qAgentLinear import QLearningAgentLinear
from main import train_agent

taxi_env = gym.make("Taxi-v3").env

linear_agent = QLearningAgentLinear(taxi_env.observation_space.n, taxi_env.action_space.n, taxi_env)

cProfile.run('train_agent(linear_agent, taxi_env, 1)')