import asyncio  # WICHTIG: Erlaubt dem Browser zu "atmen"
import sys
import numpy as np
import pygame


# Die Logik- und Grafik-Imports aus deinen Unterordnern
from game_logic import GameState, Lander, Meteor, MAX_FUEL, WORLD_WIDTH, WORLD_HEIGHT, METEOR_MAX_RADIUS, spawn_meteor
from rendering import draw_screen

# Globale Konfigurationen
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


# --- REINES NUMPY GEHIRN FÜR DEN BROWSER (OHNE TORCH) ---
class NumPyDQN:
    def __init__(self):
        # Lädt die dauerhaft gespeicherten .npy Dateien aus dem Hauptordner
        self.w0 = np.load("net_0_weight.npy")
        self.b0 = np.load("net_0_bias.npy")
        self.w2 = np.load("net_2_weight.npy")
        self.b2 = np.load("net_2_bias.npy")
        self.w4 = np.load("net_4_weight.npy")
        self.b4 = np.load("net_4_bias.npy")

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Layer 1: Dot-Product (w0 * x) + Bias, danach ReLU
        x = self.relu(np.dot(self.w0, x) + self.b0)
        
        # Layer 2: Dot-Product (w2 * x) + Bias, danach ReLU
        x = self.relu(np.dot(self.w2, x) + self.b2)
        
        # Output Layer: Reines Dot-Product (w4 * x) + Bias (ohne Aktivierungsfunktion)
        q_values = np.dot(self.w4, x) + self.b4
        return q_values


def get_observation(game_state: GameState) -> np.ndarray:
    lander = game_state.lander
    meteor = game_state.meteor
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
        (lander.x - pad_center_x) / WORLD_WIDTH,
        meteor.x / WORLD_WIDTH,
        meteor.y / WORLD_HEIGHT,
        meteor.vx / VELOCITY_NORM,
        meteor.vy / VELOCITY_NORM,
        meteor.radius / METEOR_MAX_RADIUS
    ], dtype=np.float32)


# Nutzt jetzt die NumPyDQN-Klasse statt PyTorch-Tensoren
def get_ai_action(model: NumPyDQN, game_state: GameState) -> int:
    observation = get_observation(game_state)
    q_values = model.forward(observation)
    return int(np.argmax(q_values))


def is_round_over(game_state: GameState) -> bool:
    lander = game_state.lander
    return lander.has_landed or not lander.is_alive


def reset_round(ai_state: GameState, human_state: GameState) -> None:
    ai_state.randomize_pad()
    pad_x_start = ai_state.pad_x_start
    pad_x_end = ai_state.pad_x_end

    meteor_template = spawn_meteor()

    for state in (ai_state, human_state):
        state.pad_x_start = pad_x_start
        state.pad_x_end = pad_x_end
        state.lander = Lander(WORLD_WIDTH // 2, 100)
        state.flight_time = 0.0
        state.meteor = Meteor(
            meteor_template.x,
            meteor_template.y,
            meteor_template.vx,
            meteor_template.vy,
            meteor_template.radius
        )


def sync_meteor_respawn(ai_state: GameState, human_state: GameState, last_meteor_id: int) -> int:
    if id(ai_state.meteor) != last_meteor_id:
        human_state.meteor = Meteor(
            ai_state.meteor.x,
            ai_state.meteor.y,
            ai_state.meteor.vx,
            ai_state.meteor.vy,
            ai_state.meteor.radius
        )
        return id(ai_state.meteor)
    return last_meteor_id


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


# Hauptschleife als asynchrone Funktion deklariert
async def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.RESIZABLE
    )
    pygame.display.set_caption("Lunar Lander - AI vs Human")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    header_font = pygame.font.SysFont(None, 18)

    # Initialisiere das leichtgewichtige NumPy-Netzwerk
    q_net = NumPyDQN()

    ai_state = GameState()
    human_state = GameState()
    reset_round(ai_state, human_state)
    last_meteor_id = id(ai_state.meteor)

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

        # --- HYBRIDE EINGABE-ABFRAGE (PC & MOBILE TOUCH) ---
        keys = pygame.key.get_pressed()
        left_rect, right_rect = get_side_rects(screen)
        round_over = is_round_over(ai_state) and is_round_over(human_state)
        
        # 1. PC Standard-Tastaturbelegung
        human_main_thrust = keys[pygame.K_UP] or keys[pygame.K_SPACE]
        human_rotate_left = keys[pygame.K_LEFT]
        human_rotate_right = keys[pygame.K_RIGHT]
        
        # 2. Touch-Steuerung abfragen (falls der Bildschirm berührt/geklickt wird)
        if pygame.mouse.get_pressed()[0]:
            touch_x, touch_y = pygame.mouse.get_pos()
            
            # Überprüfen, ob im Header getippt wurde (um Runden neuzustarten)
            if round_over and touch_y <= HEADER_HEIGHT:
                reset_round(ai_state, human_state)
                last_meteor_id = id(ai_state.meteor)
                round_over = False
            
            # Überprüfen auf Berührungen in der menschlichen (rechten) Bildschirmhälfte
            elif touch_x > right_rect.left and touch_y > HEADER_HEIGHT:
                relative_x = touch_x - right_rect.left
                zone_width = right_rect.width // 3
                
                if relative_x < zone_width:
                    human_rotate_left = True     # Linkes Drittel der rechten Seite
                elif relative_x > zone_width * 2:
                    human_rotate_right = True    # Rechtes Drittel der rechten Seite
                else:
                    human_main_thrust = True     # Mittleres Drittel der rechten Seite

        # Tastatur-Reset bei Rundenende
        if keys[pygame.K_r] and round_over:
            reset_round(ai_state, human_state)
            last_meteor_id = id(ai_state.meteor)
            round_over = False

        # KI-Aktionen abrufen
        if not is_round_over(ai_state):
            ai_action = get_ai_action(q_net, ai_state)

        # Inputs an die States übermitteln
        ai_state.set_inputs(
            main_thrust=ai_action == 1,
            rotate_right=ai_action == 2,
            rotate_left=ai_action == 3
        )

        human_state.set_inputs(
            main_thrust=human_main_thrust,
            rotate_right=human_rotate_right,
            rotate_left=human_rotate_left
        )

        # Physik-Updates
        while accumulator >= PHYSICS_DT:
            if not round_over:
                ai_state.update(PHYSICS_DT)
                human_state.update(PHYSICS_DT)
                last_meteor_id = sync_meteor_respawn(ai_state, human_state, last_meteor_id)
            accumulator -= PHYSICS_DT

        # Rendering vorbereiten
        left_surface = get_side_surface(side_surfaces, "left", left_rect.size)
        right_surface = get_side_surface(side_surfaces, "right", right_rect.size)

        draw_screen(left_surface, font, header_font, ai_state)
        draw_screen(right_surface, font, header_font, human_state)

        screen.fill(BLACK)
        screen.blit(left_surface, left_rect.topleft)
        screen.blit(right_surface, right_rect.topleft)
        draw_header(screen, header_font, left_rect, right_rect)

        if round_over:
            message = header_font.render("Beide Seiten fertig - R (PC) oder Header (Handy) für neue Runde", True, WHITE)
            screen.blit(message, message.get_rect(center=(screen.get_width() // 2, HEADER_HEIGHT // 2)))

        pygame.display.flip()
        
        # WICHTIG FÜR PYGBAG: Übergibt in jedem Frame kurz die Kontrolle an den Browser
        await asyncio.sleep(0)

    pygame.quit()


# Startbefehl via asyncio
if __name__ == "__main__":
    asyncio.run(main())