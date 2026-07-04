import pygame
import math
from game_logic import WORLD_WIDTH, WORLD_HEIGHT

WHITE = (255, 255, 255)
GREY = (100, 100, 100)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


def get_viewport(screen: pygame.Surface) -> tuple[float, int, int]:
    """
    Berechnet den sichtbaren Viewport für das Spielfeld.

    Der Viewport behält das Seitenverhältnis der Logikwelt bei.
    Bereiche außerhalb des Viewports bleiben als Rand sichtbar.

    Args:
        screen (pygame.Surface): Aktuelles Pygame-Fenster.

    Returns:
        tuple[float, int, int]:
            scale: Skalierungsfaktor von Worldspace zu Screenspace.
            offset_x: Horizontale Verschiebung des Viewports.
            offset_y: Vertikale Verschiebung des Viewports.
    """
    screen_width, screen_height = screen.get_size()

    scale = min(
        screen_width / WORLD_WIDTH,
        screen_height / WORLD_HEIGHT
    )

    viewport_width = int(WORLD_WIDTH * scale)
    viewport_height = int(WORLD_HEIGHT * scale)

    offset_x = (screen_width - viewport_width) // 2
    offset_y = (screen_height - viewport_height) // 2

    return scale, offset_x, offset_y


def world_to_screen(x: float, y: float, screen: pygame.Surface) -> tuple[int, int]:
    """
    Wandelt eine Worldspace-Koordinate in eine Screenspace-Koordinate um.

    Args:
        x (float): X-Koordinate im Worldspace.
        y (float): Y-Koordinate im Worldspace.
        screen (pygame.Surface): Aktuelles Pygame-Fenster.

    Returns:
        tuple[int, int]: Umgerechnete X- und Y-Koordinate im Screenspace.
    """
    scale, offset_x, offset_y = get_viewport(screen)
    return int(offset_x + x * scale), int(offset_y + y * scale)


def scale_length(value: float, screen: pygame.Surface) -> int:
    """
    Skaliert eine Länge aus dem Worldspace in den Screenspace.

    Args:
        value (float): Länge im Worldspace.
        screen (pygame.Surface): Aktuelles Pygame-Fenster.

    Returns:
        int: Skalierte Länge im Screenspace.
    """
    scale, _, _ = get_viewport(screen)
    return max(1, int(value * scale))


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: float,
    y: float,
    color: tuple[int, int, int]
) -> None:
    """
    Zeichnet Text an einer bestimmten Worldspace-Koordinate.

    Args:
        screen (pygame.Surface): Oberfläche, auf die gezeichnet wird.
        font (pygame.font.Font): Schriftart für den Text.
        text (str): Darzustellender Text.
        x (float): X-Koordinate im Worldspace.
        y (float): Y-Koordinate im Worldspace.
        color (tuple[int, int, int]): Textfarbe als RGB-Tupel.

    Returns:
        None
    """
    sx, sy = world_to_screen(x, y, screen)
    screen.blit(font.render(text, True, color), (sx, sy))


def draw_centered_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int]
) -> None:
    """
    Zeichnet Text zentriert in der Mitte des Spielfeld-Viewports.

    Args:
        screen (pygame.Surface): Oberfläche, auf die gezeichnet wird.
        font (pygame.font.Font): Schriftart für den Text.
        text (str): Darzustellender Text.
        color (tuple[int, int, int]): Textfarbe als RGB-Tupel.

    Returns:
        None
    """
    scale, offset_x, offset_y = get_viewport(screen)

    viewport_width = int(WORLD_WIDTH * scale)
    viewport_height = int(WORLD_HEIGHT * scale)

    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(
        center=(
            offset_x + viewport_width // 2,
            offset_y + viewport_height // 2
        )
    )

    screen.blit(text_surface, text_rect)


def draw_lander(screen: pygame.Surface, lander) -> None:
    """
    Zeichnet den Lander inklusive Rotation und Triebwerksflamme.

    Args:
        screen (pygame.Surface): Oberfläche, auf die gezeichnet wird.
        lander: Lander-Objekt mit Position, Geschwindigkeit, Winkel und Zustand.

    Returns:
        None
    """
    points = [
        (0, -15),
        (-10, 15),
        (10, 15)
    ]

    rotated_points = []
    rad = math.radians(lander.angle)

    for px, py in points:
        rx = px * math.cos(rad) - py * math.sin(rad)
        ry = px * math.sin(rad) + py * math.cos(rad)

        rotated_points.append(
            world_to_screen(lander.x + rx, lander.y + ry, screen)
        )

    color = GREEN if lander.has_landed else (RED if not lander.is_alive else WHITE)
    pygame.draw.polygon(screen, color, rotated_points, scale_length(2, screen))

    if lander.main_thrust_on and lander.fuel > 0 and lander.is_alive and not lander.has_landed:
        flame_points = [(-5, 16), (0, 28), (5, 16)]
        rotated_flame = []

        for fx, fy in flame_points:
            rx = fx * math.cos(rad) - fy * math.sin(rad)
            ry = fx * math.sin(rad) + fy * math.cos(rad)

            rotated_flame.append(
                world_to_screen(lander.x + rx, lander.y + ry, screen)
            )

        pygame.draw.polygon(screen, YELLOW, rotated_flame)


def draw_screen(screen, font, game_state):
    screen.fill(GREY)

    scale, offset_x, offset_y = get_viewport(screen)
    viewport_width = int(WORLD_WIDTH * scale)
    viewport_height = int(WORLD_HEIGHT * scale)

    pygame.draw.rect(
        screen,
        BLACK,
        (offset_x, offset_y, viewport_width, viewport_height)
    )

    lander = game_state.lander

    pad_start = world_to_screen(game_state.pad_x_start, game_state.pad_y, screen)
    pad_end = world_to_screen(game_state.pad_x_end, game_state.pad_y, screen)

    pygame.draw.line(
        screen,
        GREEN,
        pad_start,
        pad_end,
        scale_length(5, screen)
    )

    draw_lander(screen, lander)

    color_vx = GREEN if abs(lander.vx) < 1.0 else RED
    color_vy = GREEN if abs(lander.vy) < 1.5 else RED

    draw_text(screen, font, f"Treibstoff: {int(lander.fuel)}", 10, 10, WHITE)
    draw_text(screen, font, f"H-Geschw.: {lander.vx:.2f}", 10, 35, color_vx)
    draw_text(screen, font, f"V-Geschw.: {lander.vy:.2f}", 10, 60, color_vy)
    draw_text(screen, font, f"Winkel: {int(lander.angle)}°", 10, 85, WHITE)

    if not lander.is_alive:
        draw_centered_text(screen, font, "CRASH! Drücke R für Reset", RED)

    elif lander.has_landed:
        draw_centered_text(
            screen,
            font,
            f"ERFOLGREICHE LANDUNG! Zeit: {game_state.flight_time:.2f}s - Drücke R für Reset",
            GREEN
        )

    pygame.display.flip()