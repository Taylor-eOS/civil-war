import random
import math
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Civil War Game")
clock = pygame.time.Clock()
SOLDIER_RADIUS = 9
SHOOT_SPEED = 9
MISS_OFFSET_VALUE = 20
SHOT_EXTENSION = 100
SHOT_COOLDOWN = 1000
BASE_HIT_CHANCE = 0.5
ANIMATION_DELAY = 500
LEFT_DEFAULT_POSITIONS = [(50, i * 50 + 50) for i in range(10)]
RIGHT_DEFAULT_POSITIONS = [(750, i * 50 + 50) for i in range(10)]

class Dot:
    def __init__(self, x, y, color):
        self.pos = (x, y)
        self.color = color
        self.last_shot = 0
        self.alive = True

class Shot:
    def __init__(self, shooter, target, hit, miss_offset):
        self.shooter = shooter
        self.target = target
        self.hit = hit
        self.progress = 0
        dx = target.pos[0] - shooter.pos[0]
        dy = target.pos[1] - shooter.pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance != 0:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x, dir_y = 0, 0
        if hit:
            self.destination = target.pos
        else:
            perp_x = -dir_y
            perp_y = dir_x
            offset_direction = random.choice([-1, 1])
            self.destination = (
                target.pos[0] + dir_x * SHOT_EXTENSION + perp_x * miss_offset * offset_direction,
                target.pos[1] + dir_y * SHOT_EXTENSION + perp_y * miss_offset * offset_direction)
        total_dx = self.destination[0] - shooter.pos[0]
        total_dy = self.destination[1] - shooter.pos[1]
        self.distance = math.sqrt(total_dx**2 + total_dy**2)
        if self.distance != 0:
            self.dx = total_dx / self.distance
            self.dy = total_dy / self.distance
        else:
            self.dx, self.dy = 0, 0

    def get_current_position(self):
        return (
            self.shooter.pos[0] + self.dx * self.progress,
            self.shooter.pos[1] + self.dy * self.progress)

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
        pygame.draw.circle(screen, dot.color, dot.pos, SOLDIER_RADIUS)
    for dot in ALIVE_RIGHT_DOTS:
        pygame.draw.circle(screen, dot.color, dot.pos, SOLDIER_RADIUS)
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
                hit_chance = 0.5
                if shooter_side == 'right':
                    hit_chance = BASE_HIT_CHANCE * (left_count / right_count)
                else:
                    hit_chance = BASE_HIT_CHANCE * (right_count / left_count)
                hit_chance = min(hit_chance, 1.0)
                print(f'{left_count} {right_count}')
                #print(f'hit chance before distance calculation {hit_chance}')
                distance = ((target.pos[0] - shooter.pos[0]) ** 2 + (target.pos[1] - shooter.pos[1]) ** 2) ** 0.5
                #print(f'distance {distance}')
                normalized_distance = (distance - 700) / (32.1658488546619 - 700)
                #print(f'normalized distance {normalized_distance}')
                accuracy_modifier = 1 - 0.2 * normalized_distance #Reduces up to 20% at the max distance
                #print(f'accuracy modifier {accuracy_modifier}')
                hit_chance = hit_chance * accuracy_modifier
                hit_chance = min(hit_chance, 1.0)
                print(f'final hit chance {hit_chance}')
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
    print("It's a tie!")
pygame.quit()

