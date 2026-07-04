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


class GameState:
    def __init__(self):
        self.pad_width = 100
        self.pad_y = 550
        self.randomize_pad()
        self.lander = Lander(WORLD_WIDTH // 2, 100)
        self.flight_time = 0.0

    def randomize_pad(self):
        self.pad_x_start = random.randint(0, WORLD_WIDTH - self.pad_width)
        self.pad_x_end = self.pad_x_start + self.pad_width

    def reset(self):
        self.randomize_pad()
        self.lander = Lander(WORLD_WIDTH // 2, 100)
        self.flight_time = 0.0

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
            else:
                lander.is_alive = False

        if lander.x < 0 or lander.x > WORLD_WIDTH or lander.y < 0:
            lander.is_alive = False