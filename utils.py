import math
import random

SHOT_EXTENSION = 100

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

