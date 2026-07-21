import math
import random
import numpy as np
import pygame

from game_logic import (
    MAX_FUEL, GameState, WORLD_WIDTH, WORLD_HEIGHT, METEOR_MAX_RADIUS, LANDER_COLLISION_RADIUS,
    SAFE_LANDING_VX, SAFE_LANDING_VY, SAFE_LANDING_ANGLE, METEOR_SAFE_MARGIN
)
from HumanGame.rendering import draw_screen

PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS
VELOCITY_NORM = 300.0

# Gewichtung der Meteor-Ausweich-Strafe (Sicherheitsabstand METEOR_SAFE_MARGIN kommt aus game_logic)
METEOR_AVOIDANCE_WEIGHT = 20.0

# Belohnung für aktives Ausweichen (Vergrößerung der Meteor-Distanz), solange der Sicherheitsabstand
# unterschritten ist. Stärker gewichtet als die reine Nähe-Strafe, damit sich schnelles Ausweichen
# trotz des Zeitaufwands klar lohnt und nicht nur die reine Nähe bestraft wird.
METEOR_EVASION_REWARD_WEIGHT = 40.0

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
MAX_ANGLE_PENALTY_WEIGHT = 6.0
ANGLE_DISTANCE_SCALE = 0.4

# Zusätzliche Strafe für zu schnelles Sinken (vy über target_vy), die ebenfalls mit der Nähe
# zur Plattform skaliert -> weit weg kaum Zusatzstrafe, nah an der Plattform hohe Zusatzstrafe,
# da zu schnelles Sinken dort das größte Crash-Risiko darstellt.
MAX_VY_OVERSPEED_PENALTY_WEIGHT = 6.0

# Neben der Plattform (außerhalb pad_x_start/pad_x_end) soll der Lander bei sinkender Resthöhe zur
# Absturzlinie (pad_y - 15) lieber nochmal hochfliegen statt seitlich abzustürzen. OFF_PAD_HEIGHT_SCALE
# steuert, wie schnell diese Dringlichkeit mit der Resthöhe abfällt; MAX_OFF_PAD_CLIMB_WEIGHT die Stärke.
OFF_PAD_HEIGHT_SCALE = 80.0
MAX_OFF_PAD_CLIMB_WEIGHT = 6.0

# Horizontaler Rücklenk-Term: neben der Plattform soll der Lander (mit derselben Dringlichkeit wie
# die Höhenkorrektur) aktiv Richtung Pad-Mitte steuern, statt nach einem Meteor-Ausweichmanöver
# zufällig seitlich zu landen. Wirkt wie MAX_OFF_PAD_CLIMB_WEIGHT über landing_focus gedämpft,
# tritt also erst wieder voll in Kraft, sobald die Meteor-Gefahr vorbei ist.
MAX_OFF_PAD_RETURN_WEIGHT = 6.0

# "Weiche Wand" an den Weltgrenzen (links, rechts, oben - y<0), damit der Lander nicht beim
# Meteor-Ausweichen einfach über den Bildschirmrand hinausfliegt. Wirkt unabhängig von landing_focus,
# damit sie auch während eines Ausweichmanövers voll bestehen bleibt.
BOUNDARY_SAFE_MARGIN = 100.0
MAX_BOUNDARY_PENALTY_WEIGHT = 15.0

# Zusätzliche, mit sinkender Höhe (speed_distance) skalierende Verstärkung der horizontalen
# Ausrichtungsstrafe: seitliches Abweichen von der Plattform-Mitte wird kurz vor dem Aufsetzen
# deutlich teurer als weiter oben, damit der Lander eher zentriert aufsetzt statt daneben.
MAX_DX_LANDING_PENALTY_WEIGHT = 10.0

# Beim Verfehlen der Landungs-Schwellen (vy/vx/Winkel über SAFE_LANDING_*, aber innerhalb der
# Plattform) wird die feste Absturz-Strafe abhängig davon abgemildert, wie knapp die Schwellen
# verfehlt wurden - ein knapper Fehlversuch soll spürbar weniger bestraft werden als ein grober.
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

        # Gefahr durch den Meteor, sobald der Sicherheitsabstand unterschritten wird (0 = sicher, steigt darüber hinaus an)
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

        # Landung und Ausweichen werden multiplikativ verschränkt: je näher der Meteor, desto stärker
        # wird das Landeverhalten zurückgedrängt, sodass in Gefahr nur noch das Ausweichen zählt.
        # Quadratischer statt linearer Abfall: der Auslöseradius (danger > 0) bleibt gleich, aber
        # der Pad-Zug wird innerhalb der Zone deutlich schneller zurückgedrängt.
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

        # Lander soll langsam und gerade bleiben.
        # Ziel-Geschwindigkeit läuft asymptotisch gegen den (leicht abgesenkten) Landungs-Threshold,
        # je näher der Lander der Plattform kommt, und nähert sich MAX_APPROACH_SPEED für große Distanzen an.
        # Sobald der Lander horizontal über der Plattform ist, zählt dafür nur noch die reine Höhe
        # über der Plattform (nicht die kombinierte 2D-Distanz), damit die Ziel-Geschwindigkeit dort
        # wirklich gegen den Threshold geht. Horizontale und vertikale Geschwindigkeit werden getrennt
        # bewertet, da für die Landung unterschiedliche Schwellwerte gelten (SAFE_LANDING_VX/VY).
        over_pad = self.game_state.pad_x_start + 10 <= lander.x <= self.game_state.pad_x_end - 10

        if over_pad:
            speed_distance = max(0.0, self.game_state.pad_y - lander.y) / WORLD_HEIGHT
        else:
            speed_distance = current_distance

        approach_factor = 1 - math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        target_vy = TARGET_LANDING_VY + (MAX_APPROACH_SPEED - TARGET_LANDING_VY) * approach_factor
        target_vx = TARGET_LANDING_VX + (MAX_APPROACH_SPEED - TARGET_LANDING_VX) * approach_factor

        # Direkt über der Plattform gibt es (anders als bei vx, das je nach Seite der Plattform in
        # beide Richtungen zeigen kann) nur eine erwünschte vy-Richtung: sinken. Vorzeichenunabhängig
        # verglichen (abs(vy)) könnte der Lander sonst durch Steigen mit exakt target_vy-Betrag die
        # Strafe umgehen und endlos über der Plattform schweben/wieder an Höhe gewinnen, statt zu
        # landen. Neben der Plattform bleibt der Vergleich vorzeichenunabhängig, da dort die separate
        # Off-Pad-Höhenkorrektur bewusst zum Steigen ermutigen soll.
        if over_pad:
            vy_target_error = abs(lander.vy - target_vy)
        else:
            vy_target_error = abs(abs(lander.vy) - target_vy)

        landing_reward -= vy_target_error / 100.0
        landing_reward -= abs(abs(lander.vx) - target_vx) / 100.0

        # Zusätzliche, mit sinkender Höhe skalierende Verstärkung der horizontalen Ausrichtungsstrafe
        # (oben in dx*3.0 bereits konstant enthalten) - kurz vor dem Aufsetzen wird seitliches
        # Abweichen von der Plattform-Mitte deutlich teurer, um Landungen neben der Plattform zu vermeiden.
        dx_landing_penalty_weight = MAX_DX_LANDING_PENALTY_WEIGHT * math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        landing_reward -= abs(dx) * dx_landing_penalty_weight

        # Zusätzliche, mit der Nähe zur Plattform skalierende Strafe für zu schnelles Sinken
        # (nur der Anteil oberhalb von target_vy, da zu langsames Sinken ungefährlich ist)
        vy_overspeed = max(0.0, abs(lander.vy) - target_vy)
        vy_overspeed_weight = MAX_VY_OVERSPEED_PENALTY_WEIGHT * math.exp(-speed_distance / SPEED_DISTANCE_SCALE)
        landing_reward -= (vy_overspeed / 100.0) * vy_overspeed_weight

        # Rotationsstrafe skaliert mit der Nähe zur Plattform (nah -> hohe Strafe, weit weg -> geringe Strafe)
        angle_penalty_weight = MAX_ANGLE_PENALTY_WEIGHT * math.exp(-current_distance / ANGLE_DISTANCE_SCALE)
        landing_reward -= angle_error * angle_penalty_weight

        # Neben der Plattform (nicht darüber) drängt eine wachsende Dringlichkeit zum Höhe-Aufholen,
        # je näher die Absturzlinie (pad_y - 15) kommt: Sinken (vy > 0) wird bestraft, Steigen (vy < 0)
        # wird belohnt, sodass der Lander lieber nochmal hochfliegt statt seitlich abzustürzen.
        if not over_pad:
            crash_line = self.game_state.pad_y - 15
            height_to_crash_line = max(0.0, crash_line - lander.y)
            off_pad_urgency = math.exp(-height_to_crash_line / OFF_PAD_HEIGHT_SCALE)
            landing_reward -= off_pad_urgency * lander.vy * MAX_OFF_PAD_CLIMB_WEIGHT / 100.0

            # Horizontaler Rücklenk-Term: mit derselben Dringlichkeit soll vx aktiv Richtung
            # Pad-Mitte zeigen (Bewegung Richtung Pad wird belohnt, Bewegung weg davon bestraft),
            # statt nur zu schweben - besonders wichtig direkt nach einem Meteor-Ausweichmanöver.
            direction_factor = -1.0 if dx * lander.vx > 0 else 1.0
            landing_reward += off_pad_urgency * direction_factor * abs(lander.vx) * MAX_OFF_PAD_RETURN_WEIGHT / 100.0

        reward = landing_reward * landing_focus

        # Meteor-Ausweich-Strafe bleibt unabhängig von landing_focus in voller Stärke bestehen
        reward -= danger * METEOR_AVOIDANCE_WEIGHT

        # Weiche Wand an den Weltgrenzen: bestraft Nähe zum linken/rechten Rand sowie zum oberen Rand
        # (y < 0 ist die einzige harte Y-Grenze), damit Ausweichmanöver nicht aus der Welt hinausführen.
        dist_to_boundary = min(lander.x, WORLD_WIDTH - lander.x, lander.y)
        boundary_danger = max(0.0, (BOUNDARY_SAFE_MARGIN - dist_to_boundary) / BOUNDARY_SAFE_MARGIN)
        reward -= boundary_danger * MAX_BOUNDARY_PENALTY_WEIGHT

        # Aktives Ausweichen wird zusätzlich belohnt, solange der Sicherheitsabstand unterschritten ist,
        # damit sich schnelles Wegfliegen trotz Zeit-/Treibstoffaufwand gegenüber der reinen Nähe-Strafe lohnt.
        # Der Bonus skaliert zusätzlich mit der Gefahr, sodass Ausweichen kurz vor der Kollision am
        # stärksten belohnt wird statt gleichmäßig über die ganze Sicherheitszone.
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
                # Wie knapp wurden die Landungs-Schwellen verfehlt? 0 = exakt an der Grenze,
                # größer = deutlicher verfehlt. Ein knapper Fehlversuch wird spürbar weniger
                # bestraft als ein grober, statt beide gleich wie einen harten Crash zu behandeln.
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