import mlx.optimizers
import pygame
from numba import jit
from network import PongRL
import mlx.core as mx
import mlx.nn as nn
import mlx
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 80, 80
BALL_RADIUS = 1
PADDLE_WIDTH, PADDLE_HEIGHT = 2, 20
PADDLE_SPEED = 1
BALL_SPEED_X, BALL_SPEED_Y = 1, 1
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GAMMA = 0.99  # Discount factor for future rewards
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
UPDATE_FREQ = 4  # Frequency of gradient accumulation before updating model weights
ACCUMULATION_STEPS = 4  # Number of mini-batches to accumulate gradients over
ENTROPY_BETA = 0.01  # Entropy regularization factor

backandforthdistance = 2 * (WIDTH - PADDLE_WIDTH - 2 * BALL_RADIUS)
print(f"{backandforthdistance=}")

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

# Clock for controlling the frame rate
clock = pygame.time.Clock()

def draw(screen, paddle_y, ball_x, ball_y):
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (0, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), BALL_RADIUS)
    pygame.display.flip()

@jit(nopython=True)
def step_game(paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, decision):
    reward = -0.01  # Small negative reward for each frame to discourage staying in one region

    if decision and paddle_y < HEIGHT - PADDLE_HEIGHT:
        paddle_y += PADDLE_SPEED
    if not decision and paddle_y > 0:
        paddle_y -= PADDLE_SPEED

    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
        ball_speed_y = -ball_speed_y

    if ball_x + BALL_RADIUS >= WIDTH:
        ball_speed_x = -ball_speed_x
        ball_x = WIDTH - PADDLE_WIDTH - BALL_RADIUS

    if ball_x - BALL_RADIUS <= PADDLE_WIDTH and paddle_y <= ball_y <= paddle_y + PADDLE_HEIGHT:
        ball_speed_x = -ball_speed_x
        ball_x = PADDLE_WIDTH + BALL_RADIUS
        reward = 1  # Positive reward for hitting the ball

    if ball_x - BALL_RADIUS <= 0:
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y
        return paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, -1  # Game over, negative reward

    return paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, reward

def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.current_size = 0

    def add(self, experience):
        if self.current_size < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.current_size % self.max_size] = experience
        self.current_size += 1

    def sample(self, batch_size):
        indices = np.random.choice(min(self.current_size, self.max_size), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

def loss_fn(model, all_xin, all_actions, all_rewards, entropy_beta=ENTROPY_BETA):
    # Normalize the input frames
    all_xin = (all_xin - mx.mean(all_xin)) / mx.std(all_xin)

    # Normalize rewards
    all_rewards = (all_rewards - mx.mean(all_rewards)) / mx.std(all_rewards)
    
    targets = all_actions  # (B,)
    logits = model(all_xin)  # (B,2)
    loss = nn.losses.cross_entropy(logits, targets)  # (B,)
    
    # Apply rewards as weights
    weighted_loss = mx.tensordot(loss, all_rewards, axes=1)
    
    # Entropy regularization
    probs = nn.softmax(logits)
    entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=1)
    entropy_loss = -entropy_beta * mx.mean(entropy)
    
    return weighted_loss + entropy_loss


if __name__ == '__main__':
    lr = 3e-3
    model = PongRL(WIDTH, HEIGHT)
    # model.load_weights('weights.safetensors')
    model.eval()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = mlx.optimizers.AdamW(learning_rate=lr)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    running = True
    paddle_y = HEIGHT // 2
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y
    old_pixel_array = pygame.surfarray.array_red(screen)
    game_len = 0
    num_games = 0
    update_counter = 0

    # Initialize gradient accumulator
    accumulated_grads = None

    while running:
        game_len += 1
        diff = pygame.surfarray.array_red(screen) - old_pixel_array
        diff = diff.flatten()[None, ...]
        old_pixel_array = pygame.surfarray.array_red(screen)
        diff = mx.array(diff)
        logits = model(diff)
        decision = model.decide(logits)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        try:
            paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, reward = step_game(
                paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y, decision
            )
            replay_buffer.add((diff, int(decision), reward))
        except TypeError:
            num_games += 1
            paddle_y = (HEIGHT - 2 * PADDLE_HEIGHT) * np.random.rand() + PADDLE_HEIGHT
            ball_x, ball_y = WIDTH // 2, (HEIGHT - 2 * BALL_RADIUS) * np.random.rand() + BALL_RADIUS
            ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y * (2 * np.random.randint(0, 2) - 1)
            game_len = 0

        if replay_buffer.current_size >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            all_diff, all_actions, all_rewards = zip(*batch)
            all_diff = mx.concatenate(all_diff)
            all_actions = mx.array(all_actions)
            all_rewards = compute_discounted_rewards(np.array(all_rewards), GAMMA)
            all_rewards = mx.array(all_rewards)
            loss, grads = loss_and_grad_fn(model, all_diff, all_actions, all_rewards)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                for key in accumulated_grads.keys():
                    # print(key,accumulated_grads[key])
                    accumulated_grads[key]['weight'] += grads[key]['weight']

            if update_counter % ACCUMULATION_STEPS == 0:
                # Update the model weights
                optimizer.update(model, accumulated_grads)
                mx.eval(model.parameters(), optimizer.state)
                model.eval()

                # Reset the gradient accumulator
                accumulated_grads = None

            update_counter += 1

        draw(screen, paddle_y, ball_x, ball_y)
        clock.tick(6000)  # Adjust frame rate as needed

    print('saving weights')
    model.save_weights('weights_chatgpt.safetensors')
    pygame.quit()
