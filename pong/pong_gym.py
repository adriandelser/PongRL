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
    # args = parse_args()
    # np.random.seed(args.seed)
    # mx.random.seed(args.seed)
    # print(gym.envs.registry.keys())

    returns = []

    env = gym.make("Pong-v4")#, max_episode_steps=500,render_mode="human") # start the OpenAI gym pong environment
    # env = gym.make("Pong-v4", render_mode="human") # start the OpenAI gym pong environment

    D = 6400
    SAVE_EVERY=200

    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=5000) #truncate after 5000

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
    rollout_buffer = RolloutBuffer()

    agent = REINFORCE(policy, optimizer)

    timestep = 0

    prev_x = None
    best_reward = 19
    try:
        num_episodes = 0
        while True:
            num_episodes+=1
            observation, _ = env.reset()
            rollout_buffer.clear()
            # i+=1
            done = False
            prev_x = None
            while not done:
                # env.render()  # Render the environment to visualize the agent's actions

                cur_x = prepro(observation)
                diff = cur_x - prev_x if prev_x is not None else mx.zeros(D)
                # mx.save('diff.npy', diff)
                # mx.save('obs.npy', cur_x)

                prev_x = cur_x
                action = agent.get_action(mx.array(diff))
                # Implement the chosen action. Get back the details from the enviroment after performing the action
                observation, reward, terminated, truncated, info = env.step(action)
                rollout_buffer.add(
                    obs=diff,
                    action=action,
                    reward=reward,
                )

                done = terminated or truncated
                timestep += 1
                if "episode" in info:
                    print(f"Timestep: {timestep}, Episodic Returns: {info['episode']['r']}")
                    returns.append(info['episode']['r'][0])
            observations = rollout_buffer.get("obs")
            actions = rollout_buffer.get("action")
            rewards = rollout_buffer.get("reward")
            # print(len(rewards))
            # print(i, mx.sum(rewards))
            rewards_to_go = mx.concatenate([compute_discounted_rewards(mx.array(reward), gamma=0.99) for reward in rewards])
            # print(rewards_to_go.tolist())
            # print(rewards.shape, rewards, '\n', rewards_to_go)
            # print(observations.shape)
            agent.update(observations, actions, rewards_to_go)

            if returns[-1] > best_reward:
                best_reward = returns[-1]
                policy.save_weights(f'best_weights_{best_reward}.safetensors')
            if num_episodes%SAVE_EVERY == 0:
                print('saving temp weights')
                policy.save_weights(f'weights_{num_episodes}.safetensors')
                print(optimizer.state)


    except KeyboardInterrupt:
        print('saving weights')
        policy.save_weights('weights.safetensors')
        np.save('returns.npy',np.array(returns))
        print(optimizer.state)



if __name__=='__main__':
    main()

