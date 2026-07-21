import pygame
import sys

from core.game_logic import GameState
from core.rendering import draw_screen

# --- FENSTER ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# --- PHYSIK-TIMING ---
PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS
MAX_FRAME_TIME = 0.25

def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.RESIZABLE
    )

    pygame.display.set_caption("Lunar Lander - Human Play")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    game_state = GameState()

    accumulator = 0.0

    running = True
    while running:
        frame_time = clock.tick(FPS) / 1000.0
        frame_time = min(frame_time, MAX_FRAME_TIME)

        accumulator += frame_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_r] and (
            not game_state.lander.is_alive or game_state.lander.has_landed
        ):
            game_state.reset()

        game_state.set_inputs(
            main_thrust=keys[pygame.K_UP] or keys[pygame.K_SPACE],
            rotate_right=keys[pygame.K_RIGHT],
            rotate_left=keys[pygame.K_LEFT]
        )

        while accumulator >= PHYSICS_DT:
            game_state.update(PHYSICS_DT)
            accumulator -= PHYSICS_DT

        draw_screen(screen, font, game_state)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()