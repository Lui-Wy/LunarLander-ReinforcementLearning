import math
import random
import numpy as np
import pygame

from core.game_logic import (
    MAX_FUEL, GameState, WORLD_WIDTH, WORLD_HEIGHT, METEOR_MAX_RADIUS, LANDER_COLLISION_RADIUS,
    SAFE_LANDING_VX, SAFE_LANDING_VY, SAFE_LANDING_ANGLE, METEOR_SAFE_MARGIN
)
from core.rendering import draw_screen

PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS
VELOCITY_NORM = 300.0

# --- METEOR-AUSWEICHEN ---
METEOR_AVOIDANCE_WEIGHT = 20.0  # Strafe bei Unterschreiten des Sicherheitsabstands

# Belohnt aktives Wegfliegen stärker als die reine Nähe-Strafe, damit Ausweichen sich lohnt
METEOR_EVASION_REWARD_WEIGHT = 40.0

# Weiche Wand an den Weltgrenzen, damit Ausweichmanöver nicht aus der Welt hinausführen
BOUNDARY_SAFE_MARGIN = 100.0
MAX_BOUNDARY_PENALTY_WEIGHT = 15.0

# --- LANDUNG: ZIEL-GESCHWINDIGKEIT ---
# Ziel-vy/vx läuft asymptotisch gegen den (leicht abgesenkten) Landungs-Threshold nah der Plattform
# und gegen MAX_APPROACH_SPEED weit weg. SAFE_LANDING_SPEED_MARGIN gibt Puffer unter dem echten Limit.
MAX_APPROACH_SPEED = 150.0
SPEED_DISTANCE_SCALE = 0.3
SAFE_LANDING_SPEED_MARGIN = 0.8
TARGET_LANDING_VY = SAFE_LANDING_SPEED_MARGIN * SAFE_LANDING_VY
TARGET_LANDING_VX = SAFE_LANDING_SPEED_MARGIN * SAFE_LANDING_VX

# Extra-Strafe nur für den Anteil oberhalb von target_vy (zu schnelles Sinken = größtes Crash-Risiko)
MAX_VY_OVERSPEED_PENALTY_WEIGHT = 6.0

# --- LANDUNG: ROTATION ---
# Rotationsstrafe skaliert mit Nähe zur Plattform: weit weg kaum bestraft, nah stark bestraft
MAX_ANGLE_PENALTY_WEIGHT = 6.0
ANGLE_DISTANCE_SCALE = 0.4

# --- LANDUNG: HORIZONTALE AUSRICHTUNG ---
# Verstärkt die dx-Strafe zusätzlich kurz vorm Aufsetzen, damit zentriert statt daneben gelandet wird
MAX_DX_LANDING_PENALTY_WEIGHT = 10.0

# --- LANDUNG: KORREKTUR NEBEN DER PLATTFORM ---
# Neben der Plattform lieber nochmal hochfliegen statt abzustürzen, wenn die Resthöhe knapp wird
OFF_PAD_HEIGHT_SCALE = 80.0
MAX_OFF_PAD_CLIMB_WEIGHT = 6.0

# Zusätzlich aktiv Richtung Pad-Mitte steuern statt nur zu schweben (v.a. nach Meteor-Ausweichen)
MAX_OFF_PAD_RETURN_WEIGHT = 6.0

# --- ABSTURZ-BEWERTUNG ---
# Knapp verfehlte Landungs-Schwellen werden milder bestraft als grob verfehlte (statt beide als
# harten Crash zu werten)
BAD_LANDING_NEAR_MISS_DISCOUNT = 0.6
BAD_LANDING_OVERSHOOT_SCALE = 0.5

class LunarLanderEnv:
    def __init__(self, render_mode=False):
        self.prev_distance_to_pad = None
        self.prev_meteor_distance = None
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

        lander = self.game_state.lander
        meteor = self.game_state.meteor
        self.prev_meteor_distance = math.hypot(lander.x - meteor.x, lander.y - meteor.y)

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

        # Gefahr: 0 = sicher, steigt an, sobald der Sicherheitsabstand unterschritten wird
        meteor = self.game_state.meteor
        meteor_distance = math.hypot(lander.x - meteor.x, lander.y - meteor.y)
        meteor_collision_distance = meteor.radius + LANDER_COLLISION_RADIUS
        meteor_safe_distance = meteor_collision_distance + METEOR_SAFE_MARGIN

        danger = 0.0
        if meteor_distance < meteor_safe_distance:
            danger = (meteor_safe_distance - meteor_distance) / METEOR_SAFE_MARGIN

        # Verbesserung der Meteor-Distanz (positiv = aktives Wegbewegen vom Meteor)
        meteor_distance_improvement = meteor_distance - self.prev_meteor_distance
        self.prev_meteor_distance = meteor_distance

        # Verschränkt Landung mit Ausweichen: je näher der Meteor, desto mehr tritt das Landeverhalten
        # zurück (quadratisch statt linear, damit es innerhalb der Zone schneller nachgibt)
        landing_focus = (1.0 - min(1.0, danger)) ** 2

        landing_reward = 0.0

        # Bestrafe aktuellen Abstand
        landing_reward -= current_distance * 2.0

        # Belohne Bewegung in Richtung Plattform
        landing_reward += distance_improvement * 50.0

        # zusätzlich horizontalen Fehler stärker bestrafen -> Forcieren, dass die Rakete versucht über der Plattform zu bleiben
        landing_reward -= abs(dx) * 3.0

        # Seitliche Geschwindigkeit bestrafen,
        # besonders wenn sie von der Plattform weg zeigt
        if dx * lander.vx > 0:
            landing_reward -= abs(lander.vx) / 50.0
        else:
            landing_reward += abs(lander.vx) / 100.0

        # Über der Plattform zählt für die Ziel-Geschwindigkeit nur die reine Höhe (nicht die
        # 2D-Distanz), damit sie dort wirklich gegen den Landungs-Threshold geht
        over_pad = self.game_state.pad_x_start + 10 <= lander.x <= self.game_state.pad_x_end - 10

        if over_pad:
            speed_distance = max(0.0, self.game_state.pad_y - lander.y) / WORLD_HEIGHT
        else:
            speed_distance = current_distance

        approach_factor = 1 - math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        target_vy = TARGET_LANDING_VY + (MAX_APPROACH_SPEED - TARGET_LANDING_VY) * approach_factor
        target_vx = TARGET_LANDING_VX + (MAX_APPROACH_SPEED - TARGET_LANDING_VX) * approach_factor

        # Über der Plattform nur sinken vergleichen (vorzeichenabhängig) - sonst könnte der Lander
        # durch Steigen mit Betrag = target_vy die Strafe umgehen und endlos schweben. Neben der
        # Plattform bleibt es vorzeichenunabhängig, da dort Steigen bewusst erwünscht ist.
        if over_pad:
            vy_target_error = abs(lander.vy - target_vy)
        else:
            vy_target_error = abs(abs(lander.vy) - target_vy)

        landing_reward -= vy_target_error / 100.0
        landing_reward -= abs(abs(lander.vx) - target_vx) / 100.0

        # Zusätzlich zur konstanten dx-Strafe oben: wächst kurz vorm Aufsetzen, gegen Landung daneben
        dx_landing_penalty_weight = MAX_DX_LANDING_PENALTY_WEIGHT * math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        landing_reward -= abs(dx) * dx_landing_penalty_weight

        # Nur der Anteil oberhalb von target_vy zählt - zu langsames Sinken ist ungefährlich
        vy_overspeed = max(0.0, abs(lander.vy) - target_vy)
        vy_overspeed_weight = MAX_VY_OVERSPEED_PENALTY_WEIGHT * math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        landing_reward -= (vy_overspeed / 100.0) * vy_overspeed_weight

        # Rotationsstrafe skaliert mit der Nähe zur Plattform (nah -> hohe Strafe, weit weg -> geringe Strafe)
        angle_penalty_weight = MAX_ANGLE_PENALTY_WEIGHT * math.exp(-current_distance / ANGLE_DISTANCE_SCALE)
        landing_reward -= angle_error * angle_penalty_weight

        # Neben der Plattform: je näher die Absturzlinie, desto mehr wird Sinken bestraft und
        # Steigen belohnt, damit der Lander lieber hochfliegt statt seitlich abzustürzen
        if not over_pad:
            crash_line = self.game_state.pad_y - 15
            height_to_crash_line = max(0.0, crash_line - lander.y)
            off_pad_urgency = math.exp(-height_to_crash_line / OFF_PAD_HEIGHT_SCALE)
            landing_reward -= off_pad_urgency * lander.vy * MAX_OFF_PAD_CLIMB_WEIGHT / 100.0

            # Gleiche Dringlichkeit auch horizontal: vx Richtung Pad-Mitte wird belohnt
            direction_factor = -1.0 if dx * lander.vx > 0 else 1.0
            landing_reward += off_pad_urgency * direction_factor * abs(lander.vx) * MAX_OFF_PAD_RETURN_WEIGHT / 100.0

        reward = landing_reward * landing_focus

        # Meteor-Ausweich-Strafe bleibt unabhängig von landing_focus in voller Stärke bestehen
        reward -= danger * METEOR_AVOIDANCE_WEIGHT

        # y < 0 ist die einzige harte Y-Grenze (unten regelt die Pad-Linie den Absturz)
        dist_to_boundary = min(lander.x, WORLD_WIDTH - lander.x, lander.y)
        boundary_danger = max(0.0, (BOUNDARY_SAFE_MARGIN - dist_to_boundary) / BOUNDARY_SAFE_MARGIN)
        reward -= boundary_danger * MAX_BOUNDARY_PENALTY_WEIGHT

        # Bonus skaliert mit der Gefahr, damit Ausweichen kurz vor der Kollision am meisten lohnt
        if danger > 0:
            reward += (meteor_distance_improvement / METEOR_SAFE_MARGIN) * METEOR_EVASION_REWARD_WEIGHT * (1.0 + danger)

        # nicht ewig fliegen / Treibstoff sparen
        reward -= 0.02
        if main_thrust:
            reward -= 0.01

        done = False

        if lander.has_landed:
            done = True
            reward += 300.0

        elif not lander.is_alive:
            done = True

            if lander.death_reason == "bad_landing":
                # Überschreitung der Schwellen: 0 = exakt an der Grenze, größer = deutlicher verfehlt
                vy_overshoot = max(0.0, abs(lander.vy) - SAFE_LANDING_VY) / SAFE_LANDING_VY
                vx_overshoot = max(0.0, abs(lander.vx) - SAFE_LANDING_VX) / SAFE_LANDING_VX
                angle_overshoot = max(
                    0.0, abs((lander.angle + 180) % 360 - 180) - SAFE_LANDING_ANGLE
                ) / SAFE_LANDING_ANGLE
                overshoot = vy_overshoot + vx_overshoot + angle_overshoot

                closeness = math.exp(-overshoot / BAD_LANDING_OVERSHOOT_SCALE)
                reward -= 300.0 * (1.0 - BAD_LANDING_NEAR_MISS_DISCOUNT * closeness)
            else:
                reward -= 300.0

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