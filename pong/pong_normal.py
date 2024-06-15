import mlx.optimizers
import pygame
from numba import jit
from network import PongRL
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt


# Constants
# WIDTH, HEIGHT = 800, 600
# BALL_RADIUS = 10
# PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
# PADDLE_SPEED = 5
# BALL_SPEED_X, BALL_SPEED_Y = 4, 4
WIDTH, HEIGHT = 80, 80
BALL_RADIUS = 2
PADDLE_WIDTH, PADDLE_HEIGHT = 2, 20
PADDLE_SPEED = 1
BALL_SPEED_X, BALL_SPEED_Y = 1, 1
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ENTROPY_BETA = 0.01  # Entropy regularization factor
GAMMA = 0.99  # Discount factor for future rewards
UPDATE_FREQUENCY = 20
MAX_GAME_LEN = 1000



# Initialize Pygame
pygame.init()
# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

# Clock for controlling the frame rate
clock = pygame.time.Clock()

def draw(screen, paddle_y, ball_x, ball_y):
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (0, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    # pygame.draw.rect(screen, WHITE, (WIDTH - PADDLE_WIDTH, paddle2_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), BALL_RADIUS)
    pygame.display.flip()

def step_game(paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, decision):
    reward = 0  # Small negative reward for each frame or zero since it doesn't really matter here
    if decision and paddle_y < HEIGHT - PADDLE_HEIGHT:  # pygame.K_UP
        paddle_y += PADDLE_SPEED
    if not decision and paddle_y > 0:  # pygame.K_DOWN
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
        reward = 1.0  # Positive reward for hitting the ball


    # End of game
    if ball_x - BALL_RADIUS <= 0:
        # randomize where the paddle and ball starts, along with the y velocity being up or down
        paddle_y = (HEIGHT-2*PADDLE_HEIGHT) * np.random.rand() + PADDLE_HEIGHT#HEIGHT // 2
        ball_x, ball_y = WIDTH // 2, (HEIGHT-2*BALL_RADIUS) * np.random.rand() + BALL_RADIUS
        ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y*(2*np.random.randint(0,2)-1)

        return paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, -1.0  # Game over, negative reward


    return paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, reward

def loss_fn(model, all_xin, all_actions, all_logprobs, all_rewards, entropy_beta = ENTROPY_BETA, l2_lambda = 1e-4):
    '''shapes: (B, W*H), (B,), (B,)'''
    for idx, rewards in enumerate(all_rewards):
        x_in = all_xin[idx]
        actions = all_actions[idx]
        logprobs = all_logprobs[idx]
        rewards = (rewards-mx.mean(rewards))/(mx.std(rewards)+1e-8) #(B,)
        loss = -mx.tensordot(logprobs, rewards)


    # Normalize the input frames and rewards
    # all_xin = (all_xin - mx.mean(all_xin)) / (mx.std(all_xin)+1e-8)
    all_rewards = (all_rewards-mx.mean(all_rewards))/(mx.std(all_rewards)+1e-8) #(B,)
    # print(f"{mx.std(all_xin)=}, {mx.std(all_rewards)=}")
    print(f"{all_rewards=}")
    targets = all_actions  # (B,)
    logits = model(all_xin)  # (B,2)
    print(f"{logits=}")

    loss = nn.losses.cross_entropy(logits, targets)  # (B,)

    # Apply rewards as weights
    weighted_loss = mx.tensordot(loss, all_rewards, axes=1)

    # Entropy regularization
    probs = nn.softmax(logits)
    print(f"{probs=}")
    entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=1)
    entropy_loss = -entropy_beta * mx.mean(entropy)
    # L2 regularization
    l2_loss = mx.sum(mx.array([mx.sum(param['weight'] ** 2).item() for param in model.parameters().values()])) * l2_lambda
    print(f"{weighted_loss=}, {entropy_loss=}, {l2_loss=}")
    return weighted_loss + entropy_loss + l2_loss

def loss_fn_reinforce(logprobs, rewards):
    # print(logprobs, rewards)
    # print(logprobs.shape, rewards.shape)
    loss = -logprobs@rewards
    return loss




def compute_discounted_rewards(rewards, gamma:float = GAMMA):
    #ensure rewards array is of float dtype
    discounted_rewards = mx.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

# class ReplayBuffer:
#     def __init__(self, max_size):
#         self.buffer = []
#         self.max_size = max_size
#         self.current_size = 0

#     def add(self, experience):
#         if self.current_size < self.max_size:
#             self.buffer.append(experience)
#         else:
#             self.buffer[self.current_size % self.max_size] = experience
#         self.current_size += 1

#     def sample(self, batch_size):
#         indices = np.random.choice(min(self.current_size, self.max_size), batch_size, replace=False)
#         return [self.buffer[idx] for idx in indices]

if __name__ == '__main__':
    lr = 3e-4
    model = PongRL(WIDTH, HEIGHT)
    model.load_weights('weights_100001.safetensors')
    model.eval()
    # Create the gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn_reinforce)
    # # create a mlx optimizer
    optimizer = mlx.optimizers.AdamW(learning_rate = lr)
    # replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # Main game loop
    # running = True
    # paddle_y = HEIGHT // 2
    # ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    # ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y
    # old_pixel_array = pygame.surfarray.array_red(screen)
    # all_diff = mx.zeros((1, WIDTH*HEIGHT)) #num game, nume
    # all_actions = mx.zeros((1,), dtype=mx.int32)
    # all_rewards = mx.zeros((1,), dtype=mx.float32) #this one is numpy because we need to use np.where
    # game_len = 0
    # num_games = 0
    
    running = True
    all_diff = []
    all_actions = []
    all_rewards = []
    all_logprobs = []
    # current_game_diff = mx.zeros((1, WIDTH * HEIGHT))
    # current_game_actions = mx.zeros((1,), dtype=mx.int32)
    current_game_rewards = mx.zeros((1,), dtype=mx.float32)
    current_game_logprobs = mx.zeros((1,), dtype=mx.float32)
    paddle_y = HEIGHT // 2
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y
    old_pixel_array = pygame.surfarray.array_red(screen)
    game_len = 0
    num_games = 0
    update_counter = 0
    done = False


    while running:
        game_len+=1
        diff = pygame.surfarray.array_red(screen) - old_pixel_array
        diff = diff.flatten()[None,...]
        old_pixel_array = pygame.surfarray.array_red(screen)
        diff = mx.array(diff)
        #concatenate current diff array to all diff arrays
        # all_diff = mx.concatenate((all_diff, diff))

        model.eval()
        logits = model(diff)
        decision, logprob = model.decide(logits) # up if true, down if false
        # print(decision)
        # current_game_actions = mx.concatenate((current_game_actions, mx.array([int(decision)])))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, reward = step_game(
            paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, decision
        )
        # current_game_diff = mx.concatenate((current_game_diff, diff))
        # current_game_actions = mx.concatenate((current_game_actions, mx.array([int(decision)])))
        current_game_rewards = mx.concatenate((current_game_rewards, mx.array([reward])))
        current_game_logprobs = mx.concatenate((current_game_logprobs, logprob))
        # current_game_rewards = mx.concatenate((current_game_rewards, mx.array([reward])))
        if reward == -1 or game_len>=MAX_GAME_LEN:
            # game over
            num_games+=1
            game_len=0
            print(num_games)
            discounted_rewards = compute_discounted_rewards(current_game_rewards[1:])
            all_rewards.append(discounted_rewards)
            # all_actions.append(current_game_actions[1:])
            # all_diff.append(current_game_diff[1:])
            all_logprobs.append(current_game_logprobs[1:])
            current_game_rewards = mx.zeros((1,), dtype=mx.float32)
            # current_game_actions = mx.zeros((1,), dtype=mx.int32)
            # current_game_diff = mx.zeros((1, WIDTH * HEIGHT))
            current_game_logprobs = mx.zeros((1,), dtype=mx.float32)




        if num_games%UPDATE_FREQUENCY == 0 and (reward == -1 or game_len>=MAX_GAME_LEN) and False: #reward -1 just checks that the game is over
            print("Updating model weights")
            # Update the model with the gradients. So far no computation has happened.
            print(mx.mean(mx.array([rewards.mean() for rewards in all_rewards])))
            model.train()
            for idx, rewards in enumerate(all_rewards):
                # states = all_diff[idx]
                # actions = all_actions[idx]
                # print(idx, rewards)
                logprobs = all_logprobs[idx]
                rewards = (rewards-mx.mean(rewards))/(mx.std(rewards)+1e-8) #(B,)
                loss, grads = loss_and_grad_fn(logprobs, rewards)
                optimizer.update(model, grads) 
                mx.eval(model.parameters(), optimizer.state)
            
            # reset data arrays
            # all_diff = mx.zeros((1, WIDTH*HEIGHT)) #num game, nume
            # all_actions = mx.zeros((1,), dtype=mx.int32)
            all_rewards = []
            all_logprobs = []



        draw(screen, paddle_y, ball_x, ball_y)
        clock.tick(100)
    print('saving weights')
    # model.save_weights('weights.safetensors')
    pygame.quit()
