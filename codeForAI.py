import pygame
import math
import random
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = 0.05
THRUST = 0.15
ROTATE_SPEED = 3

class LunarLanderEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Lunar Lander - Pure AI")
            self.clock = pygame.time.Clock()

        self.pad_x_start, self.pad_x_end, self.pad_y = 350, 450, 550

    def reset(self):
        self.x = float(SCREEN_WIDTH // 2 + random.randint(-50, 50))
        self.y = 100.0
        self.vx, self.vy, self.angle = 0.0, 0.0, 0.0
        self.fuel = 500
        self.is_alive, self.has_landed = True, False
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            (self.x - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2),
            (self.y / SCREEN_HEIGHT),
            self.vx / 10.0, self.vy / 10.0,
            math.radians(self.angle) / math.pi,
            self.fuel / 500.0
        ], dtype=np.float32)

    def step(self, action):
        rad = math.radians(self.angle)
        self.main_thrust_on = self.left_thrust_on = self.right_thrust_on = False

        if action == 1 and self.fuel > 0:
            self.main_thrust_on = True
            self.vx += THRUST * math.sin(rad)
            self.vy -= THRUST * math.cos(rad)
            self.fuel -= 1
        elif action == 2 and self.fuel > 0:
            self.left_thrust_on = True
            self.angle += ROTATE_SPEED
            self.fuel -= 0.2
        elif action == 3 and self.fuel > 0:
            self.right_thrust_on = True
            self.angle -= ROTATE_SPEED
            self.fuel -= 0.2

        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

        reward = -0.1
        done = False

        # Annäherungs-Belohnung
        pad_center_x = (self.pad_x_start + self.pad_x_end) / 2
        dist = math.sqrt((self.x - pad_center_x)**2 + (self.y - self.pad_y)**2)
        reward += (1.0 - (dist / SCREEN_WIDTH)) * 0.5

        if self.y >= self.pad_y - 15:
            done = True
            if self.pad_x_start <= self.x <= self.pad_x_end:
                if abs(self.vy) < 1.5 and abs(self.vx) < 1.0 and abs(self.angle % 360) < 10:
                    self.has_landed = True
                    reward += 150.0
                else:
                    self.is_alive = False
                    reward -= 100.0
            else:
                self.is_alive = False
                reward -= 100.0

        if self.x < 0 or self.x > SCREEN_WIDTH or self.y < 0:
            done = True
            reward -= 100.0

        return self._get_observation(), reward, done

    def render(self):
        if not self.render_mode: return
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (0, 255, 0), (self.pad_x_start, self.pad_y), (self.pad_x_end, self.pad_y), 5)
        points = [(0, -15), (-10, 15), (10, 15)]
        rotated = []
        rad = math.radians(self.angle)
        for px, py in points:
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            rotated.append((self.x + rx, self.y + ry))
        color = (0, 255, 0) if self.has_landed else ((255, 0, 0) if not self.is_alive else (255, 255, 255))
        pygame.draw.polygon(self.screen, color, rotated, 2)
        pygame.display.flip()
        self.clock.tick(60)