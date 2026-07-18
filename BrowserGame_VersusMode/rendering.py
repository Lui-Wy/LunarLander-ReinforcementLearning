import pygame
import math
from game_logic import WORLD_WIDTH, WORLD_HEIGHT

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


def get_viewport(screen: pygame.Surface) -> tuple[float, float, int, int]:
    screen_width, screen_height = screen.get_size()
    scale_x = screen_width / WORLD_WIDTH
    scale_y = screen_height / WORLD_HEIGHT
    return scale_x, scale_y, 0, 0


def world_to_screen(x: float, y: float, screen: pygame.Surface) -> tuple[int, int]:
    scale_x, scale_y, _, _ = get_viewport(screen)
    return int(x * scale_x), int(y * scale_y)


def scale_length(value: float, screen: pygame.Surface, horizontal: bool = True) -> int:
    scale_x, scale_y, _, _ = get_viewport(screen)
    scale = scale_x if horizontal else scale_y
    return max(1, int(value * scale))


def draw_lander(screen: pygame.Surface, lander) -> None:
    points = [(0, -15), (-10, 15), (10, 15)]
    rotated_points = []
    rad = math.radians(lander.angle)

    for px, py in points:
        rx = px * math.cos(rad) - py * math.sin(rad)
        ry = px * math.sin(rad) + py * math.cos(rad)
        rotated_points.append(world_to_screen(lander.x + rx, lander.y + ry, screen))

    color = GREEN if lander.has_landed else (RED if not lander.is_alive else WHITE)
    
    line_thickness = max(2, scale_length(2.5, screen, True))
    pygame.draw.polygon(screen, color, rotated_points, line_thickness)

    if lander.main_thrust_on and lander.fuel > 0 and lander.is_alive and not lander.has_landed:
        flame_points = [(-5, 16), (0, 28), (5, 16)]
        rotated_flame = []
        for fx, fy in flame_points:
            rx = fx * math.cos(rad) - fy * math.sin(rad)
            ry = fx * math.sin(rad) + fy * math.cos(rad)
            rotated_flame.append(world_to_screen(lander.x + rx, lander.y + ry, screen))
        pygame.draw.polygon(screen, YELLOW, rotated_flame)


def get_streak_triangle(meteor, attach_offset: float, length_factor: float):
    speed = math.hypot(meteor.vx, meteor.vy)
    backward_angle = math.atan2(meteor.vy, meteor.vx) + math.pi
    back_x, back_y = math.cos(backward_angle), math.sin(backward_angle)
    attach_angle = backward_angle + math.radians(attach_offset)

    angular_half_width = math.radians(20.0 * length_factor)
    base_angle_1 = attach_angle - angular_half_width
    base_angle_2 = attach_angle + angular_half_width

    base1 = (meteor.x + math.cos(base_angle_1) * meteor.radius, meteor.y + math.sin(base_angle_1) * meteor.radius)
    base2 = (meteor.x + math.cos(base_angle_2) * meteor.radius, meteor.y + math.sin(base_angle_2) * meteor.radius)

    length = max(8.0, speed * METEOR_STREAK_TIME_FACTOR * length_factor)
    apex_base_x = meteor.x + math.cos(attach_angle) * meteor.radius
    apex_base_y = meteor.y + math.sin(attach_angle) * meteor.radius
    tip = (apex_base_x + back_x * length, apex_base_y + back_y * length)

    return base1, base2, tip


def draw_meteor(screen: pygame.Surface, meteor) -> None:
    for angle_offset, length_factor in meteor.streaks:
        p1, p2, p3 = get_streak_triangle(meteor, angle_offset, length_factor)
        pygame.draw.polygon(
            screen,
            METEOR_STREAK_COLOR,
            [world_to_screen(*p1, screen), world_to_screen(*p2, screen), world_to_screen(*p3, screen)]
        )

    trail_len = len(meteor.trail)
    for i, (tx, ty) in enumerate(meteor.trail):
        fade = (i + 1) / (trail_len + 1)
        dot_radius = max(1, meteor.radius * fade * 0.5)
        rx = scale_length(dot_radius, screen, True)
        ry = scale_length(dot_radius, screen, False)
        dest_rect = pygame.Rect(0, 0, rx * 2, ry * 2)
        dest_rect.center = world_to_screen(tx, ty, screen)
        pygame.draw.ellipse(screen, METEOR_TRAIL_COLOR, dest_rect)

    mx = scale_length(meteor.radius, screen, True)
    my = scale_length(meteor.radius, screen, False)
    meteor_rect = pygame.Rect(0, 0, mx * 2, my * 2)
    meteor_rect.center = world_to_screen(meteor.x, meteor.y, screen)
    pygame.draw.ellipse(screen, METEOR_COLOR, meteor_rect)


def draw_controls(screen: pygame.Surface, font: pygame.font.Font, inputs: dict) -> None:
    width, height = screen.get_size()
    
    btn_w, btn_h = 180, 70
    spacing = 40
    start_y = height - btn_h - 20
    
    total_w = (btn_w * 3) + (spacing * 2)
    start_x = (width - total_w) // 2
    
    buttons = [
        ("LINKS", start_x, inputs.get("left", False)),
        ("SCHUB", start_x + btn_w + spacing, inputs.get("main", False)),
        ("RECHTS", start_x + (btn_w + spacing) * 2, inputs.get("right", False))
    ]
    
    for text, x, is_pressed in buttons:
        rect = pygame.Rect(x, start_y, btn_w, btn_h)
        
        bg_color = (120, 120, 120, 160) if is_pressed else (50, 50, 50, 100)
        border_color = (255, 255, 255, 200) if is_pressed else (150, 150, 150, 120)
        
        btn_surf = pygame.Surface((btn_w, btn_h), pygame.SRCALPHA)
        pygame.draw.rect(btn_surf, bg_color, (0, 0, btn_w, btn_h), border_radius=8)
        pygame.draw.rect(btn_surf, border_color, (0, 0, btn_w, btn_h), width=2, border_radius=8)
        
        txt_surf = font.render(text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=(btn_w // 2, btn_h // 2))
        btn_surf.blit(txt_surf, txt_rect)
        
        screen.blit(btn_surf, rect)


def draw_screen(screen, font, header_font, game_state, current_inputs=None):
    screen.fill(BLACK)
    
    pad_start = world_to_screen(game_state.pad_x_start, game_state.pad_y, screen)
    pad_end = world_to_screen(game_state.pad_x_end, game_state.pad_y, screen)
    
    pygame.draw.line(screen, GREEN, pad_start, pad_end, 5)

    draw_meteor(screen, game_state.meteor)
    draw_lander(screen, game_state.lander)

    lander = game_state.lander
    color_vx = GREEN if abs(lander.vx) < 1.0 else RED
    color_vy = GREEN if abs(lander.vy) < 1.5 else RED

    screen_width, screen_height = screen.get_size()
    
    screen.blit(font.render(f"Treibstoff {int(lander.fuel)}", True, WHITE), (10, 10))
    screen.blit(font.render(f"H-Geschw. {lander.vx:.1f}", True, color_vx), (10, 30))
    screen.blit(font.render(f"V-Geschw.:  {lander.vy:.1f}", True, color_vy), (10, 50))
    screen.blit(font.render(f"Winkel: {int(lander.angle)}°", True, WHITE), (10, 70))

    if current_inputs is not None and lander.is_alive and not lander.has_landed:
        draw_controls(screen, font, current_inputs)

    if not lander.is_alive:
        text_surf = header_font.render("CRASH!", True, RED)
        screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))
    elif lander.has_landed:
        text_surf = header_font.render(f"SUCCESS! Zeit: {game_state.flight_time:.1f}s", True, GREEN)
        screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))