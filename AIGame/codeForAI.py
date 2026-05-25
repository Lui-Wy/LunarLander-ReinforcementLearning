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

        # optional: Startposition zufällig leicht variieren
        self.game_state.lander.x = float(WORLD_WIDTH // 2 + random.randint(-50, 50))
        self.game_state.lander.y = 100.0

        return self._get_observation()

    def _get_observation(self):
        lander = self.game_state.lander

        return np.array([
            (lander.x - WORLD_WIDTH / 2) / (WORLD_WIDTH / 2),
            lander.y / WORLD_HEIGHT,
            lander.vx / 10.0,
            lander.vy / 10.0,
            math.radians(lander.angle) / math.pi,
            lander.fuel / 500.0
        ], dtype=np.float32)

    def step(self, action):
        main_thrust = action == 1
        rotate_right = action == 2
        rotate_left = action == 3

        self.game_state.set_inputs(
            main_thrust=main_thrust,
            rotate_right=rotate_right,
            rotate_left=rotate_left
        )

        self.game_state.update(PHYSICS_DT)

        lander = self.game_state.lander

        reward = -0.1
        done = False

        pad_center_x = (self.game_state.pad_x_start + self.game_state.pad_x_end) / 2
        dist = math.sqrt(
            (lander.x - pad_center_x) ** 2 +
            (lander.y - self.game_state.pad_y) ** 2
        )

        reward += (1.0 - (dist / WORLD_WIDTH))

        if lander.has_landed:
            done = True
            reward += 150.0

        elif not lander.is_alive:
            done = True
            reward -= 100.0

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