import math
import random

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
        self.distance = distance
        if distance != 0:
            self.dx = dx / distance
            self.dy = dy / distance
        else:
            self.dx = 0
            self.dy = 0
        self.miss_offset = miss_offset if not hit else 0
        if not hit:
            self.miss_offset = miss_offset * 2 if random.random() < 0.5 else -miss_offset * 2

