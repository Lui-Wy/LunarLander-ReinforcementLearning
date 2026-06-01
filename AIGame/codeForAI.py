import math
import random
import numpy as np
import pygame

from game_logic import GameState, WORLD_WIDTH, WORLD_HEIGHT
from HumanGame.rendering import draw_screen

PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS

class LunarLanderEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.game_state = GameState()

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (WORLD_WIDTH, WORLD_HEIGHT),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Lunar Lander - AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

    def reset(self):
        self.game_state.reset()

        self.game_state.lander.x = float(WORLD_WIDTH // 2 + random.randint(-50, 50))
        self.game_state.lander.y = 100.0

        self.prev_distance_to_pad = self._distance_to_pad()

        return self._get_observation()
    
    def _distance_to_pad(self):
        lander = self.game_state.lander
        pad_center_x = (self.game_state.pad_x_start + self.game_state.pad_x_end) / 2

        dx = (lander.x - pad_center_x) / WORLD_WIDTH
        dy = (lander.y - self.game_state.pad_y) / WORLD_HEIGHT

        return math.sqrt(dx * dx + dy * dy)

    def _get_observation(self):
        lander = self.game_state.lander
        pad_center_x = (self.game_state.pad_x_start + self.game_state.pad_x_end) / 2

        angle_norm = ((lander.angle + 180) % 360 - 180) / 180

        return np.array([
            lander.x / WORLD_WIDTH,
            lander.y / WORLD_HEIGHT,
            lander.vx / 300.0,
            lander.vy / 300.0,
            angle_norm,
            lander.fuel / 500.0,
            pad_center_x / WORLD_WIDTH,
            (lander.x - pad_center_x) / WORLD_WIDTH
        ], dtype=np.float32)

    def step(self, action):
        main_thrust = action == 1
        rotate_right = action == 2
        rotate_left = action == 3

        self.game_state.set_inputs(main_thrust, rotate_right, rotate_left)
        self.game_state.update(PHYSICS_DT)

        lander = self.game_state.lander

        # Berechne horizontale Distanz zur Plattform
        pad_center_x = (self.game_state.pad_x_start + self.game_state.pad_x_end) / 2
        dx = (lander.x - pad_center_x) / WORLD_WIDTH

        # Berechne Rotationsabweichung vom "aufrechten" Zustand
        angle_error = abs((lander.angle + 180) % 360 - 180) / 180

        # Berechne Annäherung an die Plattform
        current_distance = self._distance_to_pad()
        distance_improvement = self.prev_distance_to_pad - current_distance
        self.prev_distance_to_pad = current_distance

        reward = 0.0

        # Belohne Bewegung in Richtung Plattform
        reward += distance_improvement * 50.0

        # Bestrafe aktuellen Abstand
        reward -= current_distance * 2.0

        # zusätzlich horizontalen Fehler stärker bestrafen
        reward -= abs(dx) * 3.0

        # Seitliche Geschwindigkeit bestrafen,
        # besonders wenn sie von der Plattform weg zeigt
        if dx * lander.vx > 0:
            reward -= abs(lander.vx) / 50.0
        else:
            reward += abs(lander.vx) / 100.0

        # Lander soll langsam und gerade bleiben
        reward -= abs(lander.vy) / 200.0
        reward -= angle_error * 1.0

        # nicht ewig fliegen / Treibstoff sparen
        reward -= 0.02
        if main_thrust:
            reward -= 0.01

        done = False

        if lander.has_landed:
            done = True
            reward += 200.0

        elif not lander.is_alive:
            done = True
            reward -= 150.0

        return self._get_observation(), reward, done

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        draw_screen(self.screen, self.font, self.game_state)
        self.clock.tick(60)