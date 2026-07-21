import pygame
import math
from game_logic import WORLD_WIDTH, WORLD_HEIGHT, LANDER_COLLISION_RADIUS, METEOR_SAFE_MARGIN

# Globale Flag: Zeigt den Radius um den Meteor an, ab dem der Lander beeinflusst wird
# (Sicherheitsabstand, der auch im KI-Reward zum Ausweichen verwendet wird).
SHOW_METEOR_DANGER_RADIUS = False

WHITE = (255, 255, 255)
GREY = (100, 100, 100)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
METEOR_COLOR = (170, 130, 90)
METEOR_TRAIL_COLOR = (110, 85, 60)
METEOR_STREAK_COLOR = (90, 65, 45)
METEOR_STREAK_TIME_FACTOR = 0.5
METEOR_DANGER_RADIUS_COLOR = (255, 90, 90)


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


def get_streak_triangle(meteor, attach_offset: float, length_factor: float):
    """
    Berechnet die drei Eckpunkte eines Schweif-Dreiecks im Worldspace.

    Beide Basispunkte liegen exakt auf dem Meteorrand (Abstand = Radius vom
    Mittelpunkt), sodass keine Ecke seitlich über die Kugel hinausragt. Die
    Spitze zeigt für alle Dreiecke exakt entgegen der Flugrichtung. Länge und
    Basisbreite skalieren mit der Meteorgeschwindigkeit.

    Args:
        meteor: Meteor-Objekt mit Position, Geschwindigkeit und Radius.
        attach_offset (float): Winkel-Offset des Ansatzpunkts auf dem Meteorrand in Grad.
        length_factor (float): Individueller Längen-/Breitenfaktor dieses Dreiecks.

    Returns:
        tuple: Drei (x, y) Worldspace-Punkte des Dreiecks.
    """
    speed = math.hypot(meteor.vx, meteor.vy)
    backward_angle = math.atan2(meteor.vy, meteor.vx) + math.pi
    back_x, back_y = math.cos(backward_angle), math.sin(backward_angle)

    # Ansatzpunkt darf um den Meteorrand herum variieren
    attach_angle = backward_angle + math.radians(attach_offset)

    # Beide Basisecken liegen exakt auf dem Kreisumfang, symmetrisch um den Ansatzwinkel
    angular_half_width = math.radians(20.0 * length_factor)
    base_angle_1 = attach_angle - angular_half_width
    base_angle_2 = attach_angle + angular_half_width

    base1 = (
        meteor.x + math.cos(base_angle_1) * meteor.radius,
        meteor.y + math.sin(base_angle_1) * meteor.radius
    )
    base2 = (
        meteor.x + math.cos(base_angle_2) * meteor.radius,
        meteor.y + math.sin(base_angle_2) * meteor.radius
    )

    length = max(8.0, speed * METEOR_STREAK_TIME_FACTOR * length_factor)
    apex_base_x = meteor.x + math.cos(attach_angle) * meteor.radius
    apex_base_y = meteor.y + math.sin(attach_angle) * meteor.radius

    tip = (
        apex_base_x + back_x * length,
        apex_base_y + back_y * length
    )

    return base1, base2, tip


def draw_meteor(screen: pygame.Surface, meteor) -> None:
    """
    Zeichnet den Meteor als Kugel mit Schweif-Dreiecken und einem ausblendenden
    Staubstreifen, die Flugrichtung und Geschwindigkeit erkennbar machen.

    Args:
        screen (pygame.Surface): Oberfläche, auf die gezeichnet wird.
        meteor: Meteor-Objekt mit Position, Radius und Bewegungsspur.

    Returns:
        None
    """
    for angle_offset, length_factor in meteor.streaks:
        p1, p2, p3 = get_streak_triangle(meteor, angle_offset, length_factor)

        pygame.draw.polygon(
            screen,
            METEOR_STREAK_COLOR,
            [
                world_to_screen(*p1, screen),
                world_to_screen(*p2, screen),
                world_to_screen(*p3, screen)
            ]
        )

    trail_len = len(meteor.trail)

    for i, (tx, ty) in enumerate(meteor.trail):
        fade = (i + 1) / (trail_len + 1)
        dot_radius = max(1, meteor.radius * fade * 0.5)

        pygame.draw.circle(
            screen,
            METEOR_TRAIL_COLOR,
            world_to_screen(tx, ty, screen),
            scale_length(dot_radius, screen)
        )

    pygame.draw.circle(
        screen,
        METEOR_COLOR,
        world_to_screen(meteor.x, meteor.y, screen),
        scale_length(meteor.radius, screen)
    )

    if SHOW_METEOR_DANGER_RADIUS:
        danger_radius = meteor.radius + LANDER_COLLISION_RADIUS + METEOR_SAFE_MARGIN

        pygame.draw.circle(
            screen,
            METEOR_DANGER_RADIUS_COLOR,
            world_to_screen(meteor.x, meteor.y, screen),
            scale_length(danger_radius, screen),
            scale_length(2, screen)
        )


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

    draw_meteor(screen, game_state.meteor)
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