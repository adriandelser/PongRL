import mlx.optimizers
import pygame
from numba import jit
# from policy import Policy
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy, RolloutBuffer, compute_discounted_rewards, REINFORCE


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

@jit(nopython=True)
def step(pixels, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, action):
    def draw(pixels, paddle_y, ball_x, ball_y):
        pixels[:] = BLACK
        # print(paddle_y, PADDLE_HEIGHT, PADDLE_WIDTH)
        pixels[paddle_y:paddle_y + PADDLE_HEIGHT, 0:PADDLE_WIDTH] = WHITE
        for dy in range(-BALL_RADIUS, BALL_RADIUS + 1):
            for dx in range(-BALL_RADIUS, BALL_RADIUS + 1):
                if 0 <= ball_y + dy < HEIGHT and 0 <= ball_x + dx < WIDTH:
                    # print(ball_y, dy, ball_x,dx)
                    pixels[ball_y + dy, ball_x + dx] = WHITE
    reward = 1  # Small negative reward for each frame or zero since it doesn't really matter here
    if action and paddle_y < HEIGHT - PADDLE_HEIGHT:  # pygame.K_UP
        paddle_y += PADDLE_SPEED
    if not action and paddle_y > 0:  # pygame.K_DOWN
        paddle_y -= PADDLE_SPEED


    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Bounce off the ceiling or floor
    if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
        ball_speed_y = -ball_speed_y

    #Bounce off the wall
    if ball_x + BALL_RADIUS >= WIDTH:
        ball_speed_x = -ball_speed_x
        ball_x = WIDTH - PADDLE_WIDTH - BALL_RADIUS  # Move the ball out of the paddle

    # Bounce off paddle
    if ball_x - BALL_RADIUS <= PADDLE_WIDTH and paddle_y <= ball_y <= paddle_y + PADDLE_HEIGHT:
        ball_speed_x = -ball_speed_x
        ball_x = PADDLE_WIDTH + BALL_RADIUS  # Move the ball out of the paddle
        # reward = 1.0  # Positive reward for hitting the ball


    # End of game
    if ball_x - BALL_RADIUS <= 0:
        # randomize where the paddle and ball starts, along with the y velocity being up or down
        paddle_y = int((HEIGHT-2*PADDLE_HEIGHT) * np.random.rand() + PADDLE_HEIGHT)#HEIGHT // 2
        ball_x, ball_y = WIDTH // 2, int((HEIGHT-2*BALL_RADIUS) * np.random.rand() + BALL_RADIUS)
        ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y*(2*np.random.randint(0,2)-1)
        draw(pixels, paddle_y, ball_x, ball_y)
        return pixels, reward, True, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y  # Game over, negative reward

    draw(pixels, paddle_y, ball_x, ball_y)
    return pixels, reward, False, paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y




def main():
    # args = parse_args()

    # np.random.seed(args.seed)
    # mx.random.seed(args.seed)

    # env = gym.make(
    #     id=args.env_id,
    #     render_mode=args.render,
    # )
    # env = gym.wrappers.RecordEpisodeStatistics(env)
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
        while timestep < 5000000:
            i+=1
            obs = mx.flatten(diff)
            done = False
            rollout_buffer.clear()
            while not done:
                action = agent.get_action(mx.array(obs))
                old_pixels[:,:] = pixels[:,:]
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

