import math
import random
import argparse
import pygame
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from model import TacticalGNN
from utils import Dot, Shot, get_euclid

left_default_positions = [(50, i * 50 + 50) for i in range(10)]
right_default_positions = [(750, i * 50 + 50) for i in range(10)]
max_distance = get_euclid(700, 450)
soldier_radius = 9
shoot_speed = 9
base_hit_chance = 0.5
distance_effect = 0.4
miss_offset_value = 20
animation_delay = 420
move_units = 10
full_accuracy = 300

class GameState:
    def __init__(self, left_dots=None, right_dots=None):
        self.left_dots = left_dots if left_dots is not None else []
        self.right_dots = right_dots if right_dots is not None else []
        self.all_dots = self.left_dots + self.right_dots

    def to_graph_data(self, storming_side=None):
        if not self.all_dots:
            return None
        MAX_UNITS = 20
        lcx, lcy, lspread = self._centroid_and_spread(self.left_dots)
        rcx, rcy, rspread = self._centroid_and_spread(self.right_dots)
        between = get_euclid(lcx - rcx, lcy - rcy)
        all_positions = []
        for dot in (self.left_dots + self.right_dots)[:MAX_UNITS]:
            all_positions.extend([dot.pos[0] / 800.0, dot.pos[1] / 600.0])
        while len(all_positions) < MAX_UNITS * 2:
            all_positions.append(0.0)
        node_features = []
        node_mapping = {}
        for i, dot in enumerate(self.all_dots):
            side = 1.0 if dot in self.left_dots else -1.0
            is_storming = 1.0 if (storming_side and dot in storming_side) else 0.0
            rx = (dot.pos[0] - 400) / 400.0
            ally_dots = self.left_dots if dot in self.left_dots else self.right_dots
            enemy_dots = self.right_dots if dot in self.left_dots else self.left_dots
            dists_ally = [get_euclid(dot.pos[0] - d.pos[0], dot.pos[1] - d.pos[1]) for d in ally_dots if d is not dot]
            dists_enemy = [get_euclid(dot.pos[0] - d.pos[0], dot.pos[1] - d.pos[1]) for d in enemy_dots]
            min_ally = min(dists_ally) / max_distance if dists_ally else 0.0
            avg_ally = (sum(dists_ally) / len(dists_ally)) / max_distance if dists_ally else 0.0
            min_enemy = min(dists_enemy) / max_distance if dists_enemy else 0.0
            avg_enemy = (sum(dists_enemy) / len(dists_enemy)) / max_distance if dists_enemy else 0.0
            base = [dot.pos[0] / 800.0, dot.pos[1] / 600.0, side, is_storming, len(self.left_dots) / 10.0, len(self.right_dots) / 10.0, rx]
            extras = [ lcx, lcy, lspread, rcx, rcy, rspread, between / max_distance]
            node_features.append(base + extras + [min_ally, avg_ally, min_enemy, avg_enemy] + all_positions)
            node_mapping[dot] = i
        edge_indices = []
        edge_features = []
        for i, d1 in enumerate(self.all_dots):
            for j, d2 in enumerate(self.all_dots):
                if i == j:
                    continue
                dx = d2.pos[0] - d1.pos[0]
                dy = d2.pos[1] - d1.pos[1]
                dist = get_euclid(dx, dy)
                same = 1.0 if (
                    (d1 in self.left_dots and d2 in self.left_dots) or
                    (d1 in self.right_dots and d2 in self.right_dots)) else 0.0
                edge_indices.append([i, j])
                edge_features.append([dist / max_distance, dx / 800.0, dy / 600.0, same])
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.long).t(),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32),
            'node_mapping': node_mapping}

    def _centroid_and_spread(self, dots):
        if not dots: return (0.0, 0.0, 0.0)
        xs=[d.pos[0] for d in dots]; ys=[d.pos[1] for d in dots]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        dists = [get_euclid(x - cx, y - cy) for x, y in zip(xs, ys)]
        return (cx / 800.0, cy / 600.0, sum(dists) / len(dists) / max_distance)

class GNNCivilWarGame:
    def __init__(self, model_path=None, training_mode=False):
        node_dim = 58
        edge_dim = 4
        self.model = TacticalGNN(node_dim, edge_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.training_mode = training_mode
        self.view_rect = (0, 0, 800, 600)
        if model_path:
            map_loc = None if torch.cuda.is_available() else 'cpu'
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=map_loc))
            except:
                pass
        self.reset_game()

    def reset_game(self):
        self.alive_left_dots = [Dot(x, y, (0, 0, 255)) for (x, y) in left_default_positions]
        self.alive_right_dots = [Dot(x, y, (128, 128, 128)) for (x, y) in right_default_positions]
        self.storming_side = random.choice([self.alive_left_dots, self.alive_right_dots])
        self.next_shooter_side = random.choice(['left', 'right'])
        self.game_history = []

    def get_game_state(self):
        return GameState(self.alive_left_dots, self.alive_right_dots)

    def make_decisions(self, game_state):
        graph_data = game_state.to_graph_data(self.storming_side)
        if graph_data is None:
            return None, None, None, None
        with torch.no_grad() if not self.training_mode else torch.enable_grad():
            decisions = self.model(graph_data)
        move_decisions = {}
        for i, dot in enumerate(game_state.all_dots):
            if dot in self.storming_side:
                move_decisions[dot] = torch.argmax(decisions['move_probs'][i]).item()
            else:
                move_decisions[dot] = None
        shooter_side_dots = self.alive_left_dots if self.next_shooter_side == 'left' else self.alive_right_dots
        if not shooter_side_dots:
            return move_decisions, None, None, decisions
        shooter_indices = [i for i, d in enumerate(game_state.all_dots) if d in shooter_side_dots]
        shooter_logits = decisions['shooter_logits'][shooter_indices]
        shooter_dist = torch.distributions.Categorical(logits=shooter_logits)
        rel_idx = shooter_dist.sample().item()
        shooter = shooter_side_dots[rel_idx]
        shooter_logprob = shooter_dist.log_prob(torch.tensor(rel_idx))
        target_side = self.alive_right_dots if self.next_shooter_side == 'left' else self.alive_left_dots
        if not target_side:
            return move_decisions, shooter, None, {**decisions, 'shooter_logprob': shooter_logprob}
        shooter_node_idx = game_state.all_dots.index(shooter)
        target_scores, candidates = [], []
        score_idx = 0
        for i in range(len(game_state.all_dots)):
            for j in range(len(game_state.all_dots)):
                if i != j:
                    if i == shooter_node_idx and game_state.all_dots[j] in target_side:
                        target_scores.append(decisions['target_scores'][score_idx])
                        candidates.append(game_state.all_dots[j])
                    score_idx += 1
        if target_scores:
            logits = torch.stack(target_scores)
            target_dist = torch.distributions.Categorical(logits=logits)
            targ_rel = target_dist.sample().item()
            target = candidates[targ_rel]
            target_logprob = target_dist.log_prob(torch.tensor(targ_rel))
        else:
            target = random.choice(target_side)
            target_logprob = torch.tensor(0.0)
        decisions.update(shooter_logprob=shooter_logprob, target_logprob=target_logprob)
        return move_decisions, shooter, target, decisions

    def execute_turn(self):
        if not self.alive_left_dots or not self.alive_right_dots:
            return False
        game_state = self.get_game_state()
        move_decisions, shooter, target, raw_decisions = self.make_decisions(game_state)
        if move_decisions is None:
            return False
        self.apply_movement(move_decisions)
        if shooter and target:
            self.resolve_shot(shooter, target, game_state, raw_decisions)
        return True

    def apply_movement(self, move_decisions):
        import math
        diag_step = move_units / math.sqrt(2)
        for dot, move_code in move_decisions.items():
            if dot in self.storming_side and move_code is not None:
                direction = 1 if dot in self.alive_left_dots else -1
                dx = dy = 0.0
                if move_code == 1:
                    dx = move_units * direction
                elif move_code == 2:
                    dx = diag_step * direction
                    dy = diag_step
                elif move_code == 3:
                    dx = diag_step * direction
                    dy = -diag_step
                x, y = dot.pos
                new_x = x + dx
                new_y = y + dy
                if hasattr(self, 'view_rect'):
                    x0, y0, x1, y1 = self.view_rect
                    new_x = max(min(new_x, x1 - 2), x0 + 2)
                    new_y = max(min(new_y, y1 - 2), y0 + 2)
                else:
                    new_x = max(min(new_x, 796), 4)
                    new_y = max(min(new_y, 596), 4)
                dot.pos = (new_x, new_y)

    def resolve_shot(self, shooter, target, game_state, raw_decisions):
        left_count = len(self.alive_left_dots)
        right_count = len(self.alive_right_dots)
        soldier_proportion = (right_count / left_count) if self.next_shooter_side == 'left' else (left_count / right_count)
        distance = get_euclid(target.pos[0] - shooter.pos[0], target.pos[1] - shooter.pos[1])
        normalized_distance = (distance - full_accuracy) / (max_distance - full_accuracy)
        accuracy_modifier = 1 - normalized_distance * distance_effect
        hit_chance = min(soldier_proportion * accuracy_modifier * base_hit_chance, 1.0)
        hit = random.random() < hit_chance
        if hit:
            if target in self.alive_right_dots:
                self.alive_right_dots.remove(target)
            elif target in self.alive_left_dots:
                self.alive_left_dots.remove(target)
        storming_side_name = 'left' if self.storming_side is self.alive_left_dots else 'right'
        self.game_history.append({
            'game_state': game_state,
            'decisions': raw_decisions,
            'shooter_side': self.next_shooter_side,
            'hit': hit,
            'left_count': left_count,
            'right_count': right_count,
            'storming_side_name': storming_side_name})
        self.next_shooter_side = 'right' if self.next_shooter_side == 'left' else 'left'

    def get_winner(self):
        left_alive = len(self.alive_left_dots) > 0
        right_alive = len(self.alive_right_dots) > 0
        if left_alive and not right_alive:
            return 'left'
        elif right_alive and not left_alive:
            return 'right'
        elif not left_alive and not right_alive:
            return 'tie'
        else:
            return None

    def calculate_loss(self, winner):
        total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for turn in self.game_history:
            dec = turn['decisions']
            side = turn['shooter_side']
            if winner == 'tie':
                outcome = 0.0
            elif winner == side:
                outcome = 1.0
            else:
                outcome = -1.0
            if 'shooter_logprob' in dec:
                total_loss += -dec['shooter_logprob'] * outcome
            if 'target_logprob' in dec:
                total_loss += -dec['target_logprob'] * outcome * 0.01
        return total_loss / max(len(self.game_history), 1)

    def train_step(self, winner):
        loss = self.calculate_loss(winner)
        if loss.item() != 0.0:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return loss.item()

def train_model(num_episodes=1000, model_save_path='model.pth'):
    game = GNNCivilWarGame(training_mode=True)
    win_stats = defaultdict(int)
    losses = []
    #Learning rate scheduling
    scheduler = optim.lr_scheduler.StepLR(game.optimizer, step_size=500, gamma=0.8)
    for episode in range(num_episodes):
        game.reset_game()
        #Play out the game with max turns limit
        turn_count = 0
        max_turns = 100
        while turn_count < max_turns:
            if not game.execute_turn():
                break
            turn_count += 1
        winner = game.get_winner()
        if winner:
            win_stats[winner] += 1
        else:
            #If game didn't end, call it a tie
            winner = 'tie'
            win_stats['tie'] += 1
        #Train on this episode
        loss = game.train_step(winner)
        losses.append(loss)
        scheduler.step()
        if (episode + 1) % 100 == 0:
            left_wins = win_stats['left']
            right_wins = win_stats['right']
            ties = win_stats['tie']
            total = left_wins + right_wins + ties
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Left wins: {left_wins}/{total} ({left_wins/max(total,1)*100:.1f}%)")
            print(f"Right wins: {right_wins}/{total} ({right_wins/max(total,1)*100:.1f}%)")
            print(f"Ties: {ties}/{total} ({ties/max(total,1)*100:.1f}%)")
            print(f"Average loss: {np.mean(losses[-200:]):.4f}")
            print(f"Learning rate: {game.optimizer.param_groups[0]['lr']:.6f}")
            print()
    torch.save(game.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return win_stats, losses

def apply_move(dot, move_code, move_units, left_team, right_team, view_rect=None):
    direction = 1 if dot in left_team else -1
    diag = move_units / math.sqrt(2)
    dx = dy = 0.0
    if move_code == 1:
        dx = move_units * direction
    elif move_code == 2:
        dx = diag * direction
        dy = diag
    elif move_code == 3:
        dx = diag * direction
        dy = -diag
    x, y = dot.pos
    new_x = x + dx
    new_y = y + dy
    if view_rect is not None:
        x0, y0, x1, y1 = view_rect
        new_x = max(min(new_x, x1 - 2), x0 + 2)
        new_y = max(min(new_y, y1 - 2), y0 + 2)
    else:
        new_x = max(min(new_x, 796), 4)
        new_y = max(min(new_y, 596), 4)
    dot.pos = (new_x, new_y)

def play_visual_game(model_path='model.pth'):
    pygame.init()
    pygame.display.set_caption("GNN Civil War Game")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((800, 600))
    game = GNNCivilWarGame(model_path=model_path)
    current_shot = None
    shot_in_progress = False
    last_shot_time = 0
    game_running = True
    while game_running:
        current_time = pygame.time.get_ticks()
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset_game()
                current_shot = None
                shot_in_progress = False
        for dot in game.alive_left_dots:
            pygame.draw.circle(screen, dot.color, dot.pos, soldier_radius)
        for dot in game.alive_right_dots:
            pygame.draw.circle(screen, dot.color, dot.pos, soldier_radius)
        if game.alive_left_dots and game.alive_right_dots:
            if not shot_in_progress and (current_time - last_shot_time) >= animation_delay:
                game_state = game.get_game_state()
                move_decisions, shooter, target, _ = game.make_decisions(game_state)
                if move_decisions:
                    for dot, move_code in move_decisions.items():
                        if dot in game.storming_side and move_code is not None:
                            apply_move(dot, move_code, move_units, game.alive_left_dots, game.alive_right_dots, game.view_rect)
                if shooter and target:
                    distance = get_euclid(target.pos[0] - shooter.pos[0], target.pos[1] - shooter.pos[1])
                    normalized_distance = (distance - full_accuracy) / (max_distance - full_accuracy)
                    soldier_proportion = (len(game.alive_right_dots) / len(game.alive_left_dots) if game.next_shooter_side == 'left' else len(game.alive_left_dots) / len(game.alive_right_dots))
                    accuracy_modifier = 1 - normalized_distance * distance_effect
                    hit_chance = min(soldier_proportion * accuracy_modifier * base_hit_chance, 1.0)
                    hit = random.random() < hit_chance
                    current_shot = Shot(shooter, target, hit, miss_offset_value)
                    shot_in_progress = True
                    last_shot_time = current_time
                    game.next_shooter_side = 'right' if game.next_shooter_side == 'left' else 'left'
        if shot_in_progress and current_shot:
            cx, cy = current_shot.get_current_position()
            pygame.draw.circle(screen, (0, 0, 0), (int(cx), int(cy)), 3)
            current_shot.progress += shoot_speed
            if current_shot.progress >= current_shot.distance:
                if current_shot.hit:
                    if current_shot.target in game.alive_right_dots:
                        game.alive_right_dots.remove(current_shot.target)
                    elif current_shot.target in game.alive_left_dots:
                        game.alive_left_dots.remove(current_shot.target)
                current_shot = None
                shot_in_progress = False
        pygame.display.flip()
        clock.tick(60)
    winner = game.get_winner()
    if winner == 'left':
        print("The North wins")
    elif winner == 'right':
        print("The South wins")
    else:
        print("It's a tie")
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN Civil War Game')
    parser.add_argument('--mode', choices=['train', 'play'], default='play', 
                       help='Train model or play with visualization')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    args = parser.parse_args()
    if args.mode == 'train':
        print(f"Training model for {args.episodes} episodes...")
        train_model(args.episodes, 'model.pth')
    else:
        print("Starting visual gameplay...")
        print("Press R to reset the game")
        play_visual_game('model.pth')

