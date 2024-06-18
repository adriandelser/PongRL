import mlx.optimizers
import pygame
from numba import jit
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy, RolloutBuffer, compute_discounted_rewards, REINFORCE
import gymnasium as gym
print(gym.envs.registry.keys())

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




def main():
    # args = parse_args()

    # np.random.seed(args.seed)
    # mx.random.seed(args.seed)
    print(gym.envs.registry.keys())

    env = gym.make("Pong-v0") # start the OpenAI gym pong environment
    observation_image = env.reset() # get the image
    print(observation_image.shape)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    paddle_y = HEIGHT // 2
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y

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
    pixels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    old_pixels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)

    diff = mx.zeros((HEIGHT, WIDTH), dtype=mx.int32)

    try:
        i=0
        while timestep < 500000:
            i+=1
            obs = mx.flatten(diff)
            done = False
            rollout_buffer.clear()
            while not done:
                action = agent.get_action(mx.array(obs))
                old_pixels[:,:] = pixels[:,:]
                # Implement the chosen action. Get back the details from the enviroment after performing the action
                observation_image, reward, done, info = env.step(action)
                pixels, reward, done, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y = step(pixels, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, action)
                # next_obs, reward, terminated, truncated, info = step(pixels, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, action)
                rollout_buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                )
                obs = mx.array(pixels-old_pixels).flatten()
                # mx.save('obs.npy', obs.reshape((80,80)))
                # pixels = next_pixels.copy()
                # done = terminated or truncated
                # done = True if reward == -1 else False
                timestep += 1
                # if done: in info:
                #     print(f"Timestep: {timestep}, Episodic Returns: {info['episode']['r']}")
            observations = rollout_buffer.get("obs")
            actions = rollout_buffer.get("action")
            rewards = rollout_buffer.get("reward")
            print(i, mx.sum(rewards))
            rewards_to_go = compute_discounted_rewards(rewards, gamma=0.99)
            # print(rewards.shape, rewards, '\n', rewards_to_go)
            agent.update(observations, actions, rewards_to_go)
        print('saving weights')
        policy.save_weights('weights.safetensors')

    except KeyboardInterrupt:
        print('saving weights')
        policy.save_weights('weights.safetensors')



if __name__=='__main__':
    main()

