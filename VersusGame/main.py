import sys

import numpy as np
import pygame
import torch

from game_logic import GameState, Lander, MAX_FUEL, WORLD_WIDTH, WORLD_HEIGHT
from HumanGame.rendering import draw_screen
from AIGame.trainAI import DQN

MODEL_PATH = "dataFromTraining_2.pth"

HEADER_HEIGHT = 40
SCREEN_WIDTH = WORLD_WIDTH * 2
SCREEN_HEIGHT = WORLD_HEIGHT + HEADER_HEIGHT
FPS = 60

PHYSICS_FPS = 60
PHYSICS_DT = 1.0 / PHYSICS_FPS
MAX_FRAME_TIME = 0.25

VELOCITY_NORM = 300.0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (60, 60, 60)
CYAN = (0, 200, 255)
ORANGE = (255, 165, 0)


def get_observation(game_state: GameState) -> np.ndarray:
    lander = game_state.lander
    pad_center_x = (game_state.pad_x_start + game_state.pad_x_end) / 2

    angle_norm = ((lander.angle + 180) % 360 - 180) / 180

    return np.array([
        lander.x / WORLD_WIDTH,
        lander.y / WORLD_HEIGHT,
        lander.vx / VELOCITY_NORM,
        lander.vy / VELOCITY_NORM,
        angle_norm,
        lander.fuel / MAX_FUEL,
        pad_center_x / WORLD_WIDTH,
        (lander.x - pad_center_x) / WORLD_WIDTH
    ], dtype=np.float32)


def get_ai_action(model: DQN, game_state: GameState) -> int:
    state_t = torch.FloatTensor(get_observation(game_state)).unsqueeze(0)
    with torch.no_grad():
        return model(state_t).argmax().item()


def is_round_over(game_state: GameState) -> bool:
    lander = game_state.lander
    return lander.has_landed or not lander.is_alive


def reset_round(ai_state: GameState, human_state: GameState) -> None:
    ai_state.randomize_pad()
    pad_x_start = ai_state.pad_x_start
    pad_x_end = ai_state.pad_x_end

    for state in (ai_state, human_state):
        state.pad_x_start = pad_x_start
        state.pad_x_end = pad_x_end
        state.lander = Lander(WORLD_WIDTH // 2, 100)
        state.flight_time = 0.0


def get_side_rects(screen: pygame.Surface) -> tuple[pygame.Rect, pygame.Rect]:
    width, height = screen.get_size()
    game_height = max(1, height - HEADER_HEIGHT)
    half_width = max(1, width // 2)

    left_rect = pygame.Rect(0, HEADER_HEIGHT, half_width, game_height)
    right_rect = pygame.Rect(half_width, HEADER_HEIGHT, max(1, width - half_width), game_height)

    screen_rect = screen.get_rect()
    return left_rect.clip(screen_rect), right_rect.clip(screen_rect)


def draw_header(screen: pygame.Surface, font: pygame.font.Font, left_rect: pygame.Rect, right_rect: pygame.Rect) -> None:
    pygame.draw.rect(screen, GREY, (0, 0, screen.get_width(), HEADER_HEIGHT))

    ai_label = font.render("KI", True, CYAN)
    human_label = font.render("MENSCH", True, ORANGE)

    screen.blit(ai_label, ai_label.get_rect(center=(left_rect.centerx, HEADER_HEIGHT // 2)))
    screen.blit(human_label, human_label.get_rect(center=(right_rect.centerx, HEADER_HEIGHT // 2)))

    pygame.draw.line(screen, WHITE, (left_rect.width, 0), (left_rect.width, screen.get_height()), 2)


def get_side_surface(cache: dict, key: str, size: tuple[int, int]) -> pygame.Surface:
    surface = cache.get(key)
    if surface is None or surface.get_size() != size:
        surface = pygame.Surface(size)
        cache[key] = surface
    return surface


def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.RESIZABLE
    )
    pygame.display.set_caption("Lunar Lander - AI vs Human")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    header_font = pygame.font.SysFont(None, 28)

    q_net = DQN(8, 4)
    q_net.load_state_dict(torch.load(MODEL_PATH))
    q_net.eval()

    ai_state = GameState()
    human_state = GameState()
    reset_round(ai_state, human_state)

    accumulator = 0.0
    ai_action = 0
    side_surfaces = {}

    running = True
    while running:
        frame_time = clock.tick(FPS) / 1000.0
        frame_time = min(frame_time, MAX_FRAME_TIME)
        accumulator += frame_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if (event.w, event.h) != screen.get_size():
                    screen = pygame.display.set_mode(
                        (event.w, event.h),
                        pygame.RESIZABLE
                    )

        keys = pygame.key.get_pressed()

        round_over = is_round_over(ai_state) and is_round_over(human_state)

        if keys[pygame.K_r] and round_over:
            reset_round(ai_state, human_state)
            round_over = False

        if not is_round_over(ai_state):
            ai_action = get_ai_action(q_net, ai_state)

        ai_state.set_inputs(
            main_thrust=ai_action == 1,
            rotate_right=ai_action == 2,
            rotate_left=ai_action == 3
        )

        human_state.set_inputs(
            main_thrust=keys[pygame.K_UP] or keys[pygame.K_SPACE],
            rotate_right=keys[pygame.K_RIGHT],
            rotate_left=keys[pygame.K_LEFT]
        )

        while accumulator >= PHYSICS_DT:
            if not round_over:
                ai_state.update(PHYSICS_DT)
                human_state.update(PHYSICS_DT)
            accumulator -= PHYSICS_DT

        left_rect, right_rect = get_side_rects(screen)

        left_surface = get_side_surface(side_surfaces, "left", left_rect.size)
        right_surface = get_side_surface(side_surfaces, "right", right_rect.size)

        draw_screen(left_surface, font, ai_state)
        draw_screen(right_surface, font, human_state)

        screen.fill(BLACK)
        screen.blit(left_surface, left_rect.topleft)
        screen.blit(right_surface, right_rect.topleft)
        draw_header(screen, header_font, left_rect, right_rect)

        if round_over:
            message = header_font.render("Beide Seiten fertig - R für neue Runde", True, WHITE)
            screen.blit(message, message.get_rect(center=(screen.get_width() // 2, HEADER_HEIGHT // 2)))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
