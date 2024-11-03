import pygame
import random
from collections import deque
from utils import Dot, Shot  # Assuming these are properly defined elsewhere

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("MCivil War Game")
clock = pygame.time.Clock()
DOT_RADIUS = 10
SHOOT_SPEED = 8
MISS_OFFSET_VALUE = 10
SHOT_COOLDOWN = 1000
BASE_HIT_CHANCE = 0.5
ANIMATION_DELAY = 500
LEFT_DEFAULT_POSITIONS = [(50, i * 50 + 50) for i in range(10)]
RIGHT_DEFAULT_POSITIONS = [(750, i * 50 + 50) for i in range(10)]
ALIVE_LEFT_DOTS = [Dot(x, y, (0, 0, 255)) for (x, y) in LEFT_DEFAULT_POSITIONS]
ALIVE_RIGHT_DOTS = [Dot(x, y, (128, 128, 128)) for (x, y) in RIGHT_DEFAULT_POSITIONS]
CURRENT_SHOT = None
LAST_SHOT_TIME = 0
SHOT_IN_PROGRESS = False
NEXT_SHOOTER_SIDE = random.choice(['left', 'right'])
GAME_RUNNING = True

while GAME_RUNNING:
    current_time = pygame.time.get_ticks()
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GAME_RUNNING = False
    for dot in ALIVE_LEFT_DOTS:
        pygame.draw.circle(screen, dot.color, dot.pos, DOT_RADIUS)
    for dot in ALIVE_RIGHT_DOTS:
        pygame.draw.circle(screen, dot.color, dot.pos, DOT_RADIUS)
    if ALIVE_LEFT_DOTS and ALIVE_RIGHT_DOTS:
        if not SHOT_IN_PROGRESS and (current_time - LAST_SHOT_TIME) >= ANIMATION_DELAY:
            shooter_side = NEXT_SHOOTER_SIDE
            if shooter_side == 'left':
                shooter = random.choice(ALIVE_LEFT_DOTS)
                targets = ALIVE_RIGHT_DOTS
            else:
                shooter = random.choice(ALIVE_RIGHT_DOTS)
                targets = ALIVE_LEFT_DOTS
            if targets:
                target = random.choice(targets)
                left_count = len(ALIVE_LEFT_DOTS)
                right_count = len(ALIVE_RIGHT_DOTS)
                if shooter_side == 'left':
                    difference = right_count - left_count
                else:
                    difference = left_count - right_count
                adjusted_hit_chance = BASE_HIT_CHANCE + 0.05 * max(difference, 0)
                adjusted_hit_chance = min(adjusted_hit_chance, 1.0)
                hit = random.random() < adjusted_hit_chance
                CURRENT_SHOT = Shot(shooter, target, hit, MISS_OFFSET_VALUE)
                SHOT_IN_PROGRESS = True
                LAST_SHOT_TIME = current_time
                NEXT_SHOOTER_SIDE = 'right' if shooter_side == 'left' else 'left'
        if SHOT_IN_PROGRESS and CURRENT_SHOT:
            origin = CURRENT_SHOT.shooter.pos
            target_pos = CURRENT_SHOT.target.pos
            if CURRENT_SHOT.progress == 0:
                dx = target_pos[0] - origin[0]
                dy = target_pos[1] - origin[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                CURRENT_SHOT.dx = dx / distance
                CURRENT_SHOT.dy = dy / distance
                CURRENT_SHOT.distance = distance
                if not CURRENT_SHOT.hit:
                    CURRENT_SHOT.distance = 1000
            if CURRENT_SHOT.hit and CURRENT_SHOT.target in (ALIVE_LEFT_DOTS + ALIVE_RIGHT_DOTS):
                current_x = origin[0] + CURRENT_SHOT.dx * CURRENT_SHOT.progress
                current_y = origin[1] + CURRENT_SHOT.dy * CURRENT_SHOT.progress
            else:
                if CURRENT_SHOT.progress == 0:
                    current_x = origin[0]
                    current_y = origin[1]
                else:
                    current_x = origin[0] + CURRENT_SHOT.dx * CURRENT_SHOT.progress - CURRENT_SHOT.miss_offset * CURRENT_SHOT.dy
                    current_y = origin[1] + CURRENT_SHOT.dy * CURRENT_SHOT.progress + CURRENT_SHOT.miss_offset * CURRENT_SHOT.dx
            pygame.draw.circle(screen, (0, 0, 0), (int(current_x), int(current_y)), 3)
            CURRENT_SHOT.progress += SHOOT_SPEED
            if CURRENT_SHOT.progress >= CURRENT_SHOT.distance:
                if CURRENT_SHOT.hit:
                    if CURRENT_SHOT.target in ALIVE_RIGHT_DOTS:
                        ALIVE_RIGHT_DOTS.remove(CURRENT_SHOT.target)
                    elif CURRENT_SHOT.target in ALIVE_LEFT_DOTS:
                        ALIVE_LEFT_DOTS.remove(CURRENT_SHOT.target)
                CURRENT_SHOT = None
                SHOT_IN_PROGRESS = False
    pygame.display.flip()
    clock.tick(60)

left_alive = len(ALIVE_LEFT_DOTS) > 0
right_alive = len(ALIVE_RIGHT_DOTS) > 0
if left_alive and not right_alive:
    print("Left side wins!")
elif right_alive and not left_alive:
    print("Right side wins!")
elif not left_alive and not right_alive:
    print("It's a tie!")
else:
    print("Game ended unexpectedly.")
pygame.quit()

