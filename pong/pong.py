import pygame
from pong.policy import Policy
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 210, 160
BALL_RADIUS = 3
PADDLE_WIDTH, PADDLE_HEIGHT = 5, 25
PADDLE_SPEED = 1.5
BALL_SPEED_X, BALL_SPEED_Y = 1, 1
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

# Game variables
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y
paddle1_y, paddle2_y = HEIGHT // 2, HEIGHT // 2

# Clock for controlling the frame rate
clock = pygame.time.Clock()

def draw():
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (0, paddle1_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (WIDTH - PADDLE_WIDTH, paddle2_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), BALL_RADIUS)
    pygame.display.flip()

def get_pixel_values():
    pixel_array = pygame.surfarray.array_red(screen)
    return pixel_array

def step_game(keys):
    global paddle1_y, paddle2_y, ball_x, ball_y, ball_speed_x, ball_speed_y
    if keys[pygame.K_w] and paddle1_y > 0:
        paddle1_y -= PADDLE_SPEED
    if keys[pygame.K_s] and paddle1_y < HEIGHT - PADDLE_HEIGHT:
        paddle1_y += PADDLE_SPEED
    if keys[pygame.K_UP] and paddle2_y > 0:
        paddle2_y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and paddle2_y < HEIGHT - PADDLE_HEIGHT:
        paddle2_y += PADDLE_SPEED


    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # bounce off the ceiling or floor
    if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
        ball_speed_y = -ball_speed_y

    #bounce off paddle
    if ball_x - BALL_RADIUS <= PADDLE_WIDTH and paddle1_y <= ball_y <= paddle1_y + PADDLE_HEIGHT:
        ball_speed_x = -ball_speed_x
        ball_x = PADDLE_WIDTH + BALL_RADIUS  # Move the ball out of the paddle

    if ball_x + BALL_RADIUS >= WIDTH - PADDLE_WIDTH and paddle2_y <= ball_y <= paddle2_y + PADDLE_HEIGHT:
        ball_speed_x = -ball_speed_x
        ball_x = WIDTH - PADDLE_WIDTH - BALL_RADIUS  # Move the ball out of the paddle

    #end of game
    if ball_x - BALL_RADIUS <= 0 or ball_x + BALL_RADIUS >= WIDTH:
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y

if __name__ == '__main__':
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        step_game(keys)
        draw()
        clock.tick(60)

    pygame.quit()
