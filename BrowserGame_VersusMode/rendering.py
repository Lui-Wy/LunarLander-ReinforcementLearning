import pygame
import math
import random
from game_logic import WORLD_WIDTH, WORLD_HEIGHT

# Farbpalette exakt nach der Synthwave-Vorlage
SKY_TOP = (15, 0, 35)         
SKY_MID = (50, 0, 60)        
SKY_BOT = (130, 0, 80)       

GRID_COLOR = (220, 60, 255)   
MOUNTAIN_DARK = (20, 15, 45)   

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 40, 80)          
GREEN = (0, 240, 160)        
WINDOW_BLUE = (0, 230, 255)

# Meteor-Farben
METEOR_COLOR = (90, 95, 110)
METEOR_CRATER_COLOR = (60, 65, 75)
METEOR_TRAIL_COLOR = (80, 0, 140)     
METEOR_STREAK_RGB = (0, 140, 200)     
METEOR_STREAK_TIME_FACTOR = 0.4

# Persistente Weltraum- und Bergdaten
MOUNTAIN_TRIANGLES = []
STARS = []

def get_viewport(screen: pygame.Surface) -> tuple[float, float, int, int]:
    """
    Berechnet eine einheitliche Skalierung, um Verzerrungen zu vermeiden (Aspect Ratio Lock).
    Zentriert das Spielfeld, falls das Fenster-Seitenverhältnis nicht übereinstimmt.
    """
    screen_w, screen_h = screen.get_size()
    
    # Einheitlichen Skalierungsfaktor wählen (verhindert Verzerrung)
    scale = min(screen_w / WORLD_WIDTH, screen_h / WORLD_HEIGHT)
    
    # Offsets für die Zentrierung berechnen
    offset_x = int((screen_w - (WORLD_WIDTH * scale)) / 2)
    offset_y = int((screen_h - (WORLD_HEIGHT * scale)) / 2)
    
    return scale, scale, offset_x, offset_y

def world_to_screen(x: float, y: float, screen: pygame.Surface) -> tuple[int, int]:
    scale_x, scale_y, offset_x, offset_y = get_viewport(screen)
    return int(x * scale_x + offset_x), int(y * scale_y + offset_y)

def scale_length(value: float, screen: pygame.Surface, horizontal: bool = True) -> int:
    scale_x, _, _, _ = get_viewport(screen)
    return max(1, int(value * scale_x))

def init_background_data():
    global MOUNTAIN_TRIANGLES, STARS
    if not MOUNTAIN_TRIANGLES:
        random.seed(999) 
        num_mountains = 16
        horizon_y = WORLD_HEIGHT * 0.62
        
        for _ in range(num_mountains // 2):
            peak_x = random.uniform(0, WORLD_WIDTH)
            base_w = random.uniform(80, 150)
            height = random.uniform(40, 75)
            MOUNTAIN_TRIANGLES.append((peak_x, horizon_y - height, base_w, 0))
            
        for _ in range(num_mountains // 2):
            peak_x = random.uniform(0, WORLD_WIDTH)
            base_w = random.uniform(120, 220)
            height = random.uniform(60, 105)
            MOUNTAIN_TRIANGLES.append((peak_x, horizon_y - height, base_w, 1))
            
        MOUNTAIN_TRIANGLES.sort(key=lambda m: m[3])
            
    if not STARS:
        random.seed(42)
        for _ in range(100):
            sx = random.uniform(0, WORLD_WIDTH)
            sy = random.uniform(0, WORLD_HEIGHT * 0.6)
            size = random.choice([1, 1, 2])
            brightness = random.randint(140, 255)
            STARS.append((sx, sy, size, brightness))

def draw_retro_sky(screen: pygame.Surface):
    _, _, _, offset_y = get_viewport(screen)
    width, height = screen.get_size()
    horizon_px = world_to_screen(0, WORLD_HEIGHT * 0.62, screen)[1]
    
    # Himmel-Verlauf (angepasst an den Viewport)
    for y in range(max(0, offset_y), horizon_px):
        t = (y - offset_y) / (horizon_px - offset_y) if (horizon_px - offset_y) > 0 else 0
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            factor = t / 0.5
            r = int(SKY_TOP[0] + (SKY_MID[0] - SKY_TOP[0]) * factor)
            g = int(SKY_TOP[1] + (SKY_MID[1] - SKY_TOP[1]) * factor)
            b = int(SKY_TOP[2] + (SKY_MID[2] - SKY_TOP[2]) * factor)
        else:
            factor = (t - 0.5) / 0.5
            r = int(SKY_MID[0] + (SKY_BOT[0] - SKY_MID[0]) * factor)
            g = int(SKY_MID[1] + (SKY_BOT[1] - SKY_MID[1]) * factor)
            b = int(SKY_MID[2] + (SKY_BOT[2] - SKY_MID[2]) * factor)
            
        pygame.draw.line(screen, (r, g, b), (0, y), (width, y))
        
    time_tick = pygame.time.get_ticks() * 0.002
    
    for sx, sy, size, base_bright in STARS:
        cx, cy = world_to_screen(sx, sy, screen)
        twinkle = int(base_bright * (0.7 + 0.3 * math.sin(time_tick + sx)))
        twinkle = max(0, min(255, twinkle))
        
        if size == 1:
            screen.set_at((cx, cy), (twinkle, twinkle, twinkle))
        else:
            pygame.draw.rect(screen, (twinkle, twinkle, twinkle), (cx, cy, size, size))

def draw_retro_sun(screen: pygame.Surface):
    sun_radius = scale_length(WORLD_HEIGHT * 0.22, screen)
    sun_cx, sun_cy = world_to_screen(WORLD_WIDTH / 2, WORLD_HEIGHT * 0.62 - (WORLD_HEIGHT * 0.22 * 0.3), screen)
    horizon_px = world_to_screen(0, WORLD_HEIGHT * 0.62, screen)[1]
    
    if sun_radius <= 0:
        return
        
    sun_surf = pygame.Surface((sun_radius * 2, sun_radius * 2), pygame.SRCALPHA)
    
    for sy in range(sun_radius * 2):
        dy = sy - sun_radius
        dx_max = math.sqrt(max(0, sun_radius**2 - dy**2))
        
        if dx_max > 0:
            t_color = sy / (sun_radius * 2)
            r = 255
            g = int(240 * (1.0 - t_color) + 40 * t_color)
            b = int(100 * (1.0 - t_color) + 150 * t_color)
            
            global_y = sun_cy - sun_radius + sy
            if global_y > horizon_px - sun_radius * 0.8:
                dist_to_bottom = global_y - (horizon_px - sun_radius * 0.8)
                bar_cycle = 14 + int(dist_to_bottom * 0.25)
                if (global_y % bar_cycle) < (bar_cycle * 0.35):
                    continue
            
            start_x = int(sun_radius - dx_max)
            end_x = int(sun_radius + dx_max)
            pygame.draw.line(sun_surf, (r, g, b), (start_x, sy), (end_x, sy))
            
    screen.blit(sun_surf, (sun_cx - sun_radius, sun_cy - sun_radius))

def draw_retro_mountains(screen: pygame.Surface):
    init_background_data()
    horizon_px = world_to_screen(0, WORLD_HEIGHT * 0.62, screen)[1]
    scale, _, _, _ = get_viewport(screen)
    
    for peak_x, peak_y, base_w, _ in MOUNTAIN_TRIANGLES:
        cx, cy = world_to_screen(peak_x, peak_y, screen)
        half_w = (base_w * scale) / 2
        
        p1 = (cx, cy)                  
        p2 = (cx - half_w, horizon_px)  
        p3 = (cx + half_w, horizon_px)  
        
        pygame.draw.polygon(screen, MOUNTAIN_DARK, [p1, p2, p3])
        
        lw = max(1, int(2 * scale))
        pygame.draw.line(screen, GRID_COLOR, p1, p2, lw)
        pygame.draw.line(screen, GRID_COLOR, p1, p3, lw)

def draw_retro_grid(screen: pygame.Surface):
    scale, _, offset_x, offset_y = get_viewport(screen)
    width, height = screen.get_size()
    
    horizon_px = world_to_screen(0, WORLD_HEIGHT * 0.62, screen)[1]
    bottom_px = world_to_screen(0, WORLD_HEIGHT, screen)[1]
    
    # Hintergrund-Auffüllung außerhalb des verzerren-gesicherten Grids
    pygame.draw.rect(screen, BLACK, (0, horizon_px, width, height - horizon_px))
    
    # Horizontales Glühen
    glow_surf = pygame.Surface((width, 6), pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (*GRID_COLOR, 80), (0, 0, width, 2))
    pygame.draw.rect(glow_surf, (*GRID_COLOR, 40), (0, 2, width, 4))
    screen.blit(glow_surf, (0, horizon_px - 3))
    
    # 1. Horizontale Linien (Projektion)
    num_horiz = 14
    grid_height = bottom_px - horizon_px
    for i in range(num_horiz):
        t = (i / (num_horiz - 1)) ** 2.5
        y = horizon_px + int(grid_height * t)
        pygame.draw.line(screen, GRID_COLOR, (0, y), (width, y), 1)
        
    # 2. Vertikale Gitterlinien (Fluchtpunkt liegt exakt mittig am Fuß der Berge)
    num_vert = 26
    flucht_x, flucht_y = world_to_screen(WORLD_WIDTH / 2, WORLD_HEIGHT * 0.62, screen)
    
    grid_width_world = WORLD_WIDTH * 2.4
    world_start_x = -WORLD_WIDTH * 0.7
    
    for i in range(num_vert):
        t_vert = i / (num_vert - 1)
        x_bottom_world = world_start_x + t_vert * grid_width_world
        x_bottom_px = int(x_bottom_world * scale + offset_x)
        pygame.draw.line(screen, GRID_COLOR, (flucht_x, flucht_y), (x_bottom_px, bottom_px), 1)

def draw_lander(screen: pygame.Surface, lander) -> None:
    ship_poly = [(0, -18), (8, -6), (10, 8), (14, 16), (4, 12), (-4, 12), (-14, 16), (-10, 8), (-8, -6)]
    cockpit_poly = [(0, -12), (4, -6), (-4, -6)]
    left_leg = [(-10, 8), (-16, 19), (-11, 19)]
    right_leg = [(10, 8), (16, 19), (12, 19)]

    rad = math.radians(lander.angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    scale, _, offset_x, offset_y = get_viewport(screen)

    def transform_points(pts):
        transformed = []
        for px, py in pts:
            # Rotation korrekt berechnen
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # FIX: Jetzt wird ry statt py benutzt, damit das Schiff beim Drehen stabil bleibt
            scr_x = int((lander.x + rx) * scale + offset_x)
            scr_y = int((lander.y + ry) * scale + offset_y)
            transformed.append((scr_x, scr_y))
        return transformed

    status_color = GREEN if lander.has_landed else (RED if not lander.is_alive else WHITE)

    pygame.draw.polygon(screen, (20, 10, 40), transform_points(left_leg))
    pygame.draw.polygon(screen, (20, 10, 40), transform_points(right_leg))
    pygame.draw.polygon(screen, (10, 5, 20), transform_points(ship_poly))
    pygame.draw.polygon(screen, status_color, transform_points(ship_poly), width=2)
    pygame.draw.polygon(screen, WINDOW_BLUE, transform_points(cockpit_poly))

    if lander.main_thrust_on and lander.fuel > 0 and lander.is_alive and not lander.has_landed:
        pulse = 5 * math.sin(pygame.time.get_ticks() * 0.06)
        flame_outer = [(-5, 13), (0, 30 + pulse), (5, 13)]
        pygame.draw.polygon(screen, RED, transform_points(flame_outer))
        pygame.draw.polygon(screen, WINDOW_BLUE, transform_points([(-2, 13), (0, 20 + pulse), (2, 13)]))

def draw_secure_streak(screen: pygame.Surface, meteor, angle_offset: float, length_factor: float):
    cx, cy = world_to_screen(meteor.x, meteor.y, screen)
    r_pixel = scale_length(meteor.radius, screen)
    
    speed = math.hypot(meteor.vx, meteor.vy)
    if speed == 0:
        return

    backward_angle = math.atan2(meteor.vy, meteor.vx) + math.pi
    back_x, back_y = math.cos(backward_angle), math.sin(backward_angle)
    attach_angle = backward_angle + math.radians(angle_offset)

    angular_half_width = math.radians(15.0 * length_factor)
    
    b1_x = cx + math.cos(attach_angle - angular_half_width) * r_pixel
    b1_y = cy + math.sin(attach_angle - angular_half_width) * r_pixel
    b2_x = cx + math.cos(attach_angle + angular_half_width) * r_pixel
    b2_y = cy + math.sin(attach_angle + angular_half_width) * r_pixel

    streak_len = min(r_pixel * 3.5, speed * METEOR_STREAK_TIME_FACTOR * r_pixel * length_factor)
    
    apex_x = cx + math.cos(attach_angle) * r_pixel
    apex_y = cy + math.sin(attach_angle) * r_pixel
    tip_x = apex_x + back_x * streak_len
    tip_y = apex_y + back_y * streak_len

    min_x = min(b1_x, b2_x, tip_x) - 5
    max_x = max(b1_x, b2_x, tip_x) + 5
    min_y = min(b1_y, b2_y, tip_y) - 5
    max_y = max(b1_y, b2_y, tip_y) + 5
    
    w, h = int(max_x - min_x), int(max_y - min_y)
    if w <= 0 or h <= 0:
        return
        
    streak_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    p1 = (int(b1_x - min_x), int(b1_y - min_y))
    p2 = (int(b2_x - min_x), int(b2_y - min_y))
    p3 = (int(tip_x - min_x), int(tip_y - min_y))
    
    pygame.draw.polygon(streak_surf, (*METEOR_STREAK_RGB, 120), [p1, p2, p3])
    screen.blit(streak_surf, (int(min_x), int(min_y)))

def draw_meteor(screen: pygame.Surface, meteor) -> None:
    for angle_offset, length_factor in meteor.streaks:
        draw_secure_streak(screen, meteor, angle_offset, length_factor)

    trail_len = len(meteor.trail)
    for i, (tx, ty) in enumerate(meteor.trail):
        fade = (i + 1) / (trail_len + 1)
        r_partikel = scale_length(meteor.radius * fade * 0.4, screen)
        cx_t, cy_t = world_to_screen(tx, ty, screen)
        
        p_surf = pygame.Surface((r_partikel * 2, r_partikel * 2), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*METEOR_TRAIL_COLOR, int(130 * fade)), (r_partikel, r_partikel), r_partikel)
        screen.blit(p_surf, p_surf.get_rect(center=(cx_t, cy_t)))

    r_pixel = scale_length(meteor.radius, screen)
    cx, cy = world_to_screen(meteor.x, meteor.y, screen)
    
    pygame.draw.circle(screen, METEOR_COLOR, (cx, cy), r_pixel)
    
    detail_offsets = [(-0.3, -0.2, 0.25), (0.2, 0.3, 0.2), (-0.1, 0.4, 0.15), (0.4, -0.3, 0.18)]
    for ox, oy, orad_factor in detail_offsets:
        dcx = cx + int(ox * r_pixel)
        dcy = cy + int(oy * r_pixel)
        dr = int(orad_factor * r_pixel)
        pygame.draw.circle(screen, METEOR_CRATER_COLOR, (dcx, dcy), dr)

    pygame.draw.circle(screen, BLACK, (cx, cy), r_pixel, width=2)

def draw_landing_pad(screen: pygame.Surface, game_state) -> None:
    pad_start = world_to_screen(game_state.pad_x_start, game_state.pad_y, screen)
    pad_end = world_to_screen(game_state.pad_x_end, game_state.pad_y, screen)
    
    glow_width = pad_end[0] - pad_start[0]
    if glow_width <= 0:
        return
    glow_surf = pygame.Surface((glow_width, 16), pygame.SRCALPHA)
    for h in range(16):
        alpha = int(110 * (1.0 - h / 16))
        pygame.draw.line(glow_surf, (0, 240, 160, alpha), (0, h), (glow_width, h))
    screen.blit(glow_surf, (pad_start[0], pad_start[1]))

    pygame.draw.line(screen, GREEN, pad_start, pad_end, 5)
    pygame.draw.circle(screen, WHITE, pad_start, 4)
    pygame.draw.circle(screen, WHITE, pad_end, 4)

def draw_controls(screen: pygame.Surface, font: pygame.font.Font, inputs: dict) -> None:
    width, height = screen.get_size()
    # ANPASSUNG: Höhe von 70 auf 175 erhöht (2,5x größer) für barrierefreies Drücken
    btn_w, btn_h = 180, 175
    margin = 20
    bottom_y = height - btn_h - margin
    middle_y = (height - btn_h) // 2
    
    buttons = [
        ("LINKS", margin, bottom_y, inputs.get("left", False)),
        ("SCHUB", margin, middle_y, inputs.get("main", False)),
        ("RECHTS", width - btn_w - margin, bottom_y, inputs.get("right", False)),
        ("SCHUB", width - btn_w - margin, middle_y, inputs.get("main", False))
    ]
    
    for text, x, y, is_pressed in buttons:
        rect = pygame.Rect(x, y, btn_w, btn_h)
        bg_color = (220, 60, 255, 140) if is_pressed else (15, 5, 30, 160)
        border_color = WHITE if is_pressed else GRID_COLOR
        
        btn_surf = pygame.Surface((btn_w, btn_h), pygame.SRCALPHA)
        pygame.draw.rect(btn_surf, bg_color, (0, 0, btn_w, btn_h), border_radius=8)
        pygame.draw.rect(btn_surf, border_color, (0, 0, btn_w, btn_h), width=2, border_radius=8)
        
        txt_surf = font.render(text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=(btn_w // 2, btn_h // 2))
        btn_surf.blit(txt_surf, txt_rect)
        screen.blit(btn_surf, rect)

def draw_screen(screen, font, header_font, game_state, current_inputs=None):
    init_background_data()
    
    # Bildschirmhintergrund leeren, um Ränder bei zentriertem Viewport sauber zu halten
    screen.fill(BLACK)
    
    draw_retro_sky(screen)
    draw_retro_sun(screen)
    draw_retro_mountains(screen)
    draw_retro_grid(screen)
    
    draw_landing_pad(screen, game_state)
    draw_meteor(screen, game_state.meteor)
    draw_lander(screen, game_state.lander)

    lander = game_state.lander
    color_vx = GREEN if abs(lander.vx) < 1.0 else RED
    color_vy = GREEN if abs(lander.vy) < 1.5 else RED

    def render_hud_text(text, position, color):
        shadow = font.render(text, True, BLACK)
        txt = font.render(text, True, color)
        screen.blit(shadow, (position[0]+1, position[1]+1))
        screen.blit(txt, position)

    render_hud_text(f"Treibstoff: {int(lander.fuel)}", (15, 15), WHITE)
    render_hud_text(f"H-Geschw.: {lander.vx:.1f}", (15, 38), color_vx)
    render_hud_text(f"V-Geschw.: {lander.vy:.1f}", (15, 61), color_vy)
    render_hud_text(f"Winkel: {int(lander.angle)}°", (15, 84), WHITE)

    if current_inputs is not None and lander.is_alive and not lander.has_landed:
        draw_controls(screen, font, current_inputs)

    screen_width, screen_height = screen.get_size()
    if not lander.is_alive:
        txt_surf = header_font.render("CRASH", True, RED)
        screen.blit(txt_surf, txt_surf.get_rect(center=(screen_width // 2, screen_height // 2)))
    elif lander.has_landed:
        txt_surf = header_font.render(f"ERFOLG! Zeit: ({game_state.flight_time:.1f}s)", True, GREEN)
        screen.blit(txt_surf, txt_surf.get_rect(center=(screen_width // 2, screen_height // 2)))