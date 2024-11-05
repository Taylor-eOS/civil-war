import random
import math
import pygame
from utils import Dot, Shot, get_euclid, get_minimum_distance

pygame.init()
pygame.display.set_caption("Civil War Game")
clock = pygame.time.Clock()
screen = pygame.display.set_mode((800, 600))
LEFT_DEFAULT_POSITIONS = [(50, i * 50 + 50) for i in range(10)]
RIGHT_DEFAULT_POSITIONS = [(750, i * 50 + 50) for i in range(10)]
MAX_DISTANCE = get_euclid(700, 450)
ALIVE_LEFT_DOTS = [Dot(x, y, (0, 0, 255)) for (x, y) in LEFT_DEFAULT_POSITIONS]
ALIVE_RIGHT_DOTS = [Dot(x, y, (128, 128, 128)) for (x, y) in RIGHT_DEFAULT_POSITIONS]
SOLDIER_RADIUS = 9
SHOOT_SPEED = 9
BASE_HIT_CHANCE = 0.5
DISTANCE_EFFECT = 0.4
MISS_OFFSET_VALUE = 20
SHOT_COOLDOWN = 1000
ANIMATION_DELAY = 500
CURRENT_SHOT = None
LAST_SHOT_TIME = 0
SHOT_IN_PROGRESS = False
NEXT_SHOOTER_SIDE = random.choice(['left', 'right'])
GAME_RUNNING = True
MOVE_UNITS = 11
MOVE_CHANCE = 0.75
STORMING_SIDE = random.choice([ALIVE_LEFT_DOTS, ALIVE_RIGHT_DOTS])
FULL_ACCURACY = 300

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
        if not SHOT_IN_PROGRESS and (current_time - LAST_SHOT_TIME) >= ANIMATION_DELAY:
            for dot in STORMING_SIDE:
                if random.random() < MOVE_CHANCE:
                    move_direction = 1 if dot in ALIVE_LEFT_DOTS else -1
                    new_pos_x = dot.pos[0] + MOVE_UNITS * move_direction
                    dot.pos = (new_pos_x, dot.pos[1])
            shooter_side = NEXT_SHOOTER_SIDE
            if shooter_side == 'left':
                shooter = random.choice(ALIVE_LEFT_DOTS)
                targets = ALIVE_RIGHT_DOTS
                move_direction = 1
            else:
                shooter = random.choice(ALIVE_RIGHT_DOTS)
                targets = ALIVE_LEFT_DOTS
                move_direction = -1
            #if random.random() < MOVE_CHANCE:
            #    new_pos_x = shooter.pos[0] + MOVE_UNITS * move_direction
            #    shooter.pos = (new_pos_x, shooter.pos[1])
            if targets:
                target = random.choice(targets)
                left_count = len(ALIVE_LEFT_DOTS)
                right_count = len(ALIVE_RIGHT_DOTS)
                print(f'{left_count} {right_count}')
                soldier_proportion = right_count / left_count
                if shooter_side == 'right':
                    soldier_proportion = left_count / right_count
                distance = get_euclid(target.pos[0] - shooter.pos[0], target.pos[1] - shooter.pos[1])
                print(f'distance {distance:.0f}')
                normalized_distance = (distance - FULL_ACCURACY) / (MAX_DISTANCE -FULL_ACCURACY)
                print(f'normalized distance {normalized_distance:.2f}')
                accuracy_modifier = 1 - normalized_distance * DISTANCE_EFFECT
                print(f'accuracy modifier {accuracy_modifier:.2f}')
                hit_chance = soldier_proportion * accuracy_modifier
                print(f'hit chance {hit_chance:.2f}')
                hit_chance = min(hit_chance * BASE_HIT_CHANCE, 1.0)
                print(f'final hit chance {hit_chance:.2f}')
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

