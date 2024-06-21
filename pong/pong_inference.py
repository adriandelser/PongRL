import mlx.optimizers
import pygame
from numba import jit
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from .policy import Policy, RolloutBuffer, compute_discounted_rewards, REINFORCE
# print(gym.envs.registry.keys())

# Constants
WIDTH, HEIGHT = 80, 80
BALL_RADIUS = 2
PADDLE_WIDTH, PADDLE_HEIGHT = 2, 20
PADDLE_SPEED = 1
BALL_SPEED_X, BALL_SPEED_Y = 1, 1
WHITE = 1
BLACK = 0
ENTROPY_BETA = 0.01  # Entropy regularization factor
GAMMA = 0.99  # Discount factor for future rewards
UPDATE_FREQUENCY = 20
MAX_GAME_LEN = 1000

def prepro(I:mx.array):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I = mx.where(I==144, 0, I) # erase background (background type 1)
    I = mx.where(I==109, 0, I) # erase background (background type 2)
    I = mx.where(I!=0, 1, I) # everything else (paddles, ball) just set to 1
    return I.astype(mx.float32).flatten()




def main():
    env = gym.make("Pong-v4", render_mode="human") # start the OpenAI gym pong environment

    D = 6400

    env = gym.wrappers.RecordEpisodeStatistics(env)

    policy = Policy(
        num_layers=1,
        input_dim=HEIGHT*WIDTH,
        hidden_dim=200,
        output_dim=2,
        activations=[nn.relu],
    )
    policy.load_weights('weights.safetensors')
    mx.eval(policy.parameters())

    optimizer = mlx.optimizers.AdamW(learning_rate=1e-3)

    agent = REINFORCE(policy, optimizer)

    timestep = 0

    prev_x = None


    while True:
        observation, _ = env.reset()
        done = False
        while not done:
            # env.render()  # Render the environment to visualize the agent's actions

            cur_x = prepro(observation)
            diff = cur_x - prev_x if prev_x is not None else mx.zeros(D)

            prev_x = cur_x
            # logits = policy(diff)
            # action = mx.argmax(logits).item() + 2
            action = agent.get_action(mx.array(diff))
            # print(action)
            # Implement the chosen action. Get back the details from the enviroment after performing the action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated
if __name__=='__main__':
    main()

