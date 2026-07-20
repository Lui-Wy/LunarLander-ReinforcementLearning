import math
import random
import numpy as np
import pygame

from game_logic import (
    MAX_FUEL, GameState, WORLD_WIDTH, WORLD_HEIGHT, METEOR_MAX_RADIUS, LANDER_COLLISION_RADIUS,
    SAFE_LANDING_VX, SAFE_LANDING_VY
)
from HumanGame.rendering import draw_screen

PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS
VELOCITY_NORM = 300.0

# Sicherheitsabstand zum Meteor, ab dem eine Annäherung bestraft wird
METEOR_SAFE_MARGIN = 80.0
METEOR_AVOIDANCE_WEIGHT = 4.0

# Ziel-Geschwindigkeit soll asymptotisch gegen die für eine Landung zulässige Geschwindigkeit laufen,
# je näher der Lander der Plattform kommt (statt gegen 0), und sich für große Distanzen MAX_APPROACH_SPEED annähern.
# MAX_APPROACH_SPEED: erlaubte Geschwindigkeit bei großer Entfernung (Asymptote für distance -> groß)
# SPEED_DISTANCE_SCALE: steuert, wie schnell die Ziel-Geschwindigkeit mit sinkender Distanz abfällt
# SAFE_LANDING_SPEED_MARGIN: Sicherheitsmarge unter dem Landungs-Threshold, damit kleine Abweichungen
# den Grenzwert für eine erfolgreiche Landung nicht direkt reißen
MAX_APPROACH_SPEED = 150.0
SPEED_DISTANCE_SCALE = 0.3
SAFE_LANDING_SPEED_MARGIN = 0.8
TARGET_LANDING_VY = SAFE_LANDING_SPEED_MARGIN * SAFE_LANDING_VY
TARGET_LANDING_VX = SAFE_LANDING_SPEED_MARGIN * SAFE_LANDING_VX

# Bestrafung der Rotation soll mit der Nähe zur Plattform skalieren:
# weit weg -> Rotation kaum bestraft, nah an der Plattform -> Rotation stark bestraft (stabil bleiben)
MAX_ANGLE_PENALTY_WEIGHT = 3.0
ANGLE_DISTANCE_SCALE = 0.3

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
        meteor = self.game_state.meteor
        pad_center_x = (self.game_state.pad_x_start + self.game_state.pad_x_end) / 2

        angle_norm = ((lander.angle + 180) % 360 - 180) / 180

        return np.array([
            lander.x / WORLD_WIDTH,
            lander.y / WORLD_HEIGHT,
            lander.vx / VELOCITY_NORM,
            lander.vy / VELOCITY_NORM,
            angle_norm,
            lander.fuel / MAX_FUEL,
            pad_center_x / WORLD_WIDTH,
            (lander.x - pad_center_x) / WORLD_WIDTH,
            meteor.x / WORLD_WIDTH,
            meteor.y / WORLD_HEIGHT,
            meteor.vx / VELOCITY_NORM,
            meteor.vy / VELOCITY_NORM,
            meteor.radius / METEOR_MAX_RADIUS
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
        # Bringe Rotation auf [-180|180] -> [-1|1]
        angle_error = abs((lander.angle + 180) % 360 - 180) / 180

        # Berechne Annäherung an die Plattform
        current_distance = self._distance_to_pad()
        distance_improvement = self.prev_distance_to_pad - current_distance
        self.prev_distance_to_pad = current_distance

        reward = 0.0

        # Bestrafe aktuellen Abstand
        reward -= current_distance * 2.0

        # Belohne Bewegung in Richtung Plattform
        reward += distance_improvement * 50.0

        # zusätzlich horizontalen Fehler stärker bestrafen -> Forcieren, dass die Rakete versucht über der Plattform zu bleiben
        reward -= abs(dx) * 3.0

        # Seitliche Geschwindigkeit bestrafen,
        # besonders wenn sie von der Plattform weg zeigt
        if dx * lander.vx > 0:
            reward -= abs(lander.vx) / 50.0
        else:
            reward += abs(lander.vx) / 100.0

        # Lander soll langsam und gerade bleiben.
        # Ziel-Geschwindigkeit läuft asymptotisch gegen den (leicht abgesenkten) Landungs-Threshold,
        # je näher der Lander der Plattform kommt, und nähert sich MAX_APPROACH_SPEED für große Distanzen an.
        # Sobald der Lander horizontal über der Plattform ist, zählt dafür nur noch die reine Höhe
        # über der Plattform (nicht die kombinierte 2D-Distanz), damit die Ziel-Geschwindigkeit dort
        # wirklich gegen den Threshold geht. Horizontale und vertikale Geschwindigkeit werden getrennt
        # bewertet, da für die Landung unterschiedliche Schwellwerte gelten (SAFE_LANDING_VX/VY).
        if self.game_state.pad_x_start <= lander.x <= self.game_state.pad_x_end:
            speed_distance = max(0.0, self.game_state.pad_y - lander.y) / WORLD_HEIGHT
        else:
            speed_distance = current_distance

        approach_factor = 1 - math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        target_vy = TARGET_LANDING_VY + (MAX_APPROACH_SPEED - TARGET_LANDING_VY) * approach_factor
        target_vx = TARGET_LANDING_VX + (MAX_APPROACH_SPEED - TARGET_LANDING_VX) * approach_factor

        reward -= abs(abs(lander.vy) - target_vy) / 100.0
        reward -= abs(abs(lander.vx) - target_vx) / 100.0

        # Rotationsstrafe skaliert mit der Nähe zur Plattform (nah -> hohe Strafe, weit weg -> geringe Strafe)
        angle_penalty_weight = MAX_ANGLE_PENALTY_WEIGHT * math.exp(-current_distance / ANGLE_DISTANCE_SCALE)
        reward -= angle_error * angle_penalty_weight

        # Abstand zum Meteor bestrafen, sobald er den Sicherheitsabstand unterschreitet
        meteor = self.game_state.meteor
        meteor_distance = math.hypot(lander.x - meteor.x, lander.y - meteor.y)
        meteor_collision_distance = meteor.radius + LANDER_COLLISION_RADIUS
        meteor_safe_distance = meteor_collision_distance + METEOR_SAFE_MARGIN

        if meteor_distance < meteor_safe_distance:
            danger = (meteor_safe_distance - meteor_distance) / METEOR_SAFE_MARGIN
            reward -= danger * METEOR_AVOIDANCE_WEIGHT

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