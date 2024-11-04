import random
import math
import pygame
from utils import Dot, Shot

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Civil War Game")
clock = pygame.time.Clock()
SOLDIER_RADIUS = 9
SHOOT_SPEED = 9
MISS_OFFSET_VALUE = 20
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
MOVE_UNITS = 15
MOVE_CHANCE = 0.7

while GAME_RUNNING:
    current_time = pygame.time.get_ticks()
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GAME_RUNNING = False
    for dot in ALIVE_LEFT_DOTS:
        pygame.draw.circle(screen, dot.color, dot.pos, SOLDIER_RADIUS)
    for dot in ALIVE_RIGHT_DOTS:
        pygame.draw.circle(screen, dot.color, dot.pos, SOLDIER_RADIUS)
    if ALIVE_LEFT_DOTS and ALIVE_RIGHT_DOTS:
        def get_euclid(dx, dy):
            return (dx ** 2 + dy ** 2) ** 0.5
        if not SHOT_IN_PROGRESS and (current_time - LAST_SHOT_TIME) >= ANIMATION_DELAY:
            shooter_side = NEXT_SHOOTER_SIDE
            if shooter_side == 'left':
                shooter = random.choice(ALIVE_LEFT_DOTS)
                targets = ALIVE_RIGHT_DOTS
                move_direction = 1
            else:
                shooter = random.choice(ALIVE_RIGHT_DOTS)
                targets = ALIVE_LEFT_DOTS
                move_direction = -1
            if targets:
                if random.random() < MOVE_CHANCE:
                    new_pos_x = shooter.pos[0] + MOVE_UNITS * move_direction
                    shooter.pos = (new_pos_x, shooter.pos[1])
                target = random.choice(targets)
                left_count = len(ALIVE_LEFT_DOTS)
                right_count = len(ALIVE_RIGHT_DOTS)
                hit_chance = BASE_HIT_CHANCE
                if shooter_side == 'right':
                    hit_chance = BASE_HIT_CHANCE * (left_count / right_count)
                else:
                    hit_chance = BASE_HIT_CHANCE * (right_count / left_count)
                print(f'{left_count} {right_count}')
                print(f'hit chance before distance {hit_chance}')
                distance = get_euclid(target.pos[0] - shooter.pos[0], target.pos[1] - shooter.pos[1])
                print(f'distance {distance}')
                normalized_distance = (distance - 700) / (get_euclid(700, 450) - 700)
                print(f'normalized distance {normalized_distance}')
                accuracy_modifier = 1 - 0.3 * normalized_distance
                print(f'accuracy modifier {accuracy_modifier}')
                hit_chance = hit_chance * accuracy_modifier
                print(f'final hit chance {hit_chance}')
                hit_chance = min(hit_chance, 1.0)
                hit = random.random() < hit_chance
                CURRENT_SHOT = Shot(shooter, target, hit, MISS_OFFSET_VALUE)
                SHOT_IN_PROGRESS = True
                LAST_SHOT_TIME = current_time
                NEXT_SHOOTER_SIDE = 'right' if shooter_side == 'left' else 'left'
        if SHOT_IN_PROGRESS and CURRENT_SHOT:
            current_x, current_y = CURRENT_SHOT.get_current_position()
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
    print("The North wins")
elif right_alive and not left_alive:
    print("The South wins")
elif not left_alive and not right_alive:
    print("It's a tie")
pygame.quit()

