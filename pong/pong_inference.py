import mlx.optimizers
import pygame
from numba import jit
from policy import Policy
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt


# Constants
WIDTH, HEIGHT = 80, 80
BALL_RADIUS = 2
PADDLE_WIDTH, PADDLE_HEIGHT = 2, 20
PADDLE_SPEED = 1
BALL_SPEED_X, BALL_SPEED_Y = 1, 1
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)




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




if __name__ == '__main__':
    policy = Policy(
            num_layers=1,
            input_dim=HEIGHT*WIDTH,
            hidden_dim=200,
            output_dim=2,
            activations=[nn.relu],
        )    
    policy.load_weights('weights.safetensors')
    policy.eval()
    running = True

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
    
        policy.eval()
        logits = policy(diff)
        decision = mx.random.categorical(logits)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, reward = step_game(
            paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, decision
        )
    

        draw(screen, paddle_y, ball_x, ball_y)
        clock.tick(100)
    pygame.quit()
