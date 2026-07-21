import math
import random

WORLD_WIDTH = 800
WORLD_HEIGHT = 600

GRAVITY = 180
THRUST = 500
ROTATE_SPEED = 180
MAX_FUEL = 500
FUEL_REQ_MAIN = 100
FUEL_REQ_ROTATION = 20

# Akzeptanzwerte für eine erfolgreiche Landung
SAFE_LANDING_VY = 80
SAFE_LANDING_VX = 50
SAFE_LANDING_ANGLE = 10

# Meteor: Größe/Geschwindigkeit werden bei jedem (Neu-)Spawn zufällig innerhalb dieser Grenzen bestimmt
METEOR_MIN_RADIUS = 20
METEOR_MAX_RADIUS = 45
METEOR_MIN_SPEED = 80
METEOR_MAX_SPEED = 220

# Der Meteor zielt beim Spawn auf einen zufälligen Punkt innerhalb dieses zentralen Bereichs
# (statt einer rein zufälligen Richtung), damit seine Flugbahn deutlich häufiger den Bereich
# kreuzt, in dem sich der Lander tatsächlich bewegt. Enger gefasstes Zielfenster (weiter erhöhte
# Kreuzungswahrscheinlichkeit) sorgt dafür, dass er noch zuverlässiger nahe der eigentlichen
# Lander-Flugbahn kreuzt (Geschwindigkeit bleibt unverändert).
METEOR_TARGET_MARGIN_X = 0.42
METEOR_TARGET_Y_MIN_FACTOR = 0.20
METEOR_TARGET_Y_MAX_FACTOR = 0.90

LANDER_COLLISION_RADIUS = 12

# Sicherheitsabstand zum Meteor, ab dem der Lander (im Reward bzw. visuell) beeinflusst wird
METEOR_SAFE_MARGIN = 80.0


class Lander:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0

        self.fuel = MAX_FUEL
        self.is_alive = True
        self.has_landed = False
        # Grund für das Episodenende, sobald is_alive False wird: "meteor", "bad_landing",
        # "off_pad" oder "out_of_bounds" - fürs Tracking der Erfolgs-/Absturzursachen im Training.
        self.death_reason = None

        self.main_thrust_on = False
        self.left_thrust_on = False
        self.right_thrust_on = False

    def update(self, dt: float):
        if not self.is_alive or self.has_landed:
            return

        self.vy += GRAVITY * dt
        rad = math.radians(self.angle)

        if self.main_thrust_on and self.fuel > 0:
            self.vx += THRUST * math.sin(rad) * dt
            self.vy -= THRUST * math.cos(rad) * dt
            self.fuel -= FUEL_REQ_MAIN * dt

        if self.left_thrust_on and self.fuel > 0:
            self.angle += ROTATE_SPEED * dt
            self.fuel -= FUEL_REQ_ROTATION * dt

        if self.right_thrust_on and self.fuel > 0:
            self.angle -= ROTATE_SPEED * dt
            self.fuel -= FUEL_REQ_ROTATION * dt

        self.fuel = max(0, self.fuel)
        self.x += self.vx * dt
        self.y += self.vy * dt


class Meteor:
    TRAIL_LENGTH = 20
    STREAK_COUNT = 4
    STREAK_ANGLE_SPREAD = 50
    STREAK_MIN_LENGTH_FACTOR = 0.4
    STREAK_MAX_LENGTH_FACTOR = 1.3

    def __init__(self, x, y, vx, vy, radius):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.trail = []

        # Winkel-Offset und Längenfaktor je Schweif-Dreieck, einmalig pro Meteor bestimmt
        self.streaks = [
            (
                random.uniform(-self.STREAK_ANGLE_SPREAD, self.STREAK_ANGLE_SPREAD),
                random.uniform(self.STREAK_MIN_LENGTH_FACTOR, self.STREAK_MAX_LENGTH_FACTOR)
            )
            for _ in range(self.STREAK_COUNT)
        ]

    def update(self, dt: float):
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)

        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_off_screen(self) -> bool:
        """True, sobald der Meteor die Szene komplett (inkl. Radius) verlassen hat."""
        return (
            self.x + self.radius < 0 or
            self.x - self.radius > WORLD_WIDTH or
            self.y + self.radius < 0 or
            self.y - self.radius > WORLD_HEIGHT
        )


def spawn_meteor() -> Meteor:
    """
    Erzeugt einen Meteor mit zufälliger (aber pro Runde fixer) Größe und Geschwindigkeit.

    Der Meteor spawnt an einem zufälligen Bildschirmrand, zielt aber auf einen zufälligen Punkt
    innerhalb des zentralen Flugbereichs (statt einer rein zufälligen Richtung), sodass seine
    Flugbahn deutlich häufiger den Bereich kreuzt, in dem sich der Lander tatsächlich bewegt.
    """
    radius = random.uniform(METEOR_MIN_RADIUS, METEOR_MAX_RADIUS)
    speed = random.uniform(METEOR_MIN_SPEED, METEOR_MAX_SPEED)
    edge = random.choice(("left", "right", "top", "bottom"))

    if edge == "left":
        x, y = -radius, random.uniform(0, WORLD_HEIGHT)
    elif edge == "right":
        x, y = WORLD_WIDTH + radius, random.uniform(0, WORLD_HEIGHT)
    elif edge == "top":
        x, y = random.uniform(0, WORLD_WIDTH), -radius
    else:
        x, y = random.uniform(0, WORLD_WIDTH), WORLD_HEIGHT + radius

    target_x = random.uniform(METEOR_TARGET_MARGIN_X * WORLD_WIDTH, (1 - METEOR_TARGET_MARGIN_X) * WORLD_WIDTH)
    target_y = random.uniform(METEOR_TARGET_Y_MIN_FACTOR * WORLD_HEIGHT, METEOR_TARGET_Y_MAX_FACTOR * WORLD_HEIGHT)

    dx, dy = target_x - x, target_y - y
    length = math.hypot(dx, dy)
    vx = dx / length * speed
    vy = dy / length * speed

    return Meteor(x, y, vx, vy, radius)


class GameState:
    def __init__(self):
        self.pad_width = 100
        self.pad_y = 550
        self.randomize_pad()
        self.lander = Lander(WORLD_WIDTH // 2, 100)
        self.flight_time = 0.0
        self.meteor = spawn_meteor()

    def randomize_pad(self):
        self.pad_x_start = random.randint(0, WORLD_WIDTH - self.pad_width)
        self.pad_x_end = self.pad_x_start + self.pad_width

    def reset(self):
        self.randomize_pad()
        self.lander = Lander(WORLD_WIDTH // 2, 100)
        self.flight_time = 0.0
        self.meteor = spawn_meteor()

    def set_inputs(self, main_thrust, rotate_right, rotate_left):
        lander = self.lander

        if lander.is_alive and not lander.has_landed:
            lander.main_thrust_on = main_thrust
            lander.left_thrust_on = rotate_right
            lander.right_thrust_on = rotate_left
        else:
            lander.main_thrust_on = False
            lander.left_thrust_on = False
            lander.right_thrust_on = False

    def update(self, dt: float):
        lander = self.lander

        if lander.is_alive and not lander.has_landed:
            self.flight_time += dt

        lander.update(dt)

        if lander.is_alive and not lander.has_landed:
            self.meteor.update(dt)

            dist = math.hypot(lander.x - self.meteor.x, lander.y - self.meteor.y)
            if dist < self.meteor.radius + LANDER_COLLISION_RADIUS:
                lander.is_alive = False
                if lander.death_reason is None:
                    lander.death_reason = "meteor"

            if self.meteor.is_off_screen():
                self.meteor = spawn_meteor()

        if lander.y >= self.pad_y - 15:
            if self.pad_x_start <= lander.x <= self.pad_x_end:
                if (
                    abs(lander.vy) < SAFE_LANDING_VY and
                    abs(lander.vx) < SAFE_LANDING_VX and
                    abs((lander.angle + 180) % 360 - 180) < SAFE_LANDING_ANGLE
                ):
                    lander.has_landed = True
                    lander.vy = 0
                    lander.vx = 0
                else:
                    lander.is_alive = False
                    if lander.death_reason is None:
                        lander.death_reason = "bad_landing"
            else:
                lander.is_alive = False
                if lander.death_reason is None:
                    lander.death_reason = "off_pad"

        if lander.x < 0 or lander.x > WORLD_WIDTH or lander.y < 0:
            lander.is_alive = False
            if lander.death_reason is None:
                lander.death_reason = "out_of_bounds"