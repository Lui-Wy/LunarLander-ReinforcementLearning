# pygame-script
# pygbag: width=1600, height=640, scale=dynamic
import asyncio
import sys
import numpy as np
import pygame

from game_logic import GameState, Lander, Meteor, MAX_FUEL, WORLD_WIDTH, WORLD_HEIGHT, METEOR_MAX_RADIUS, spawn_meteor
from rendering import draw_screen

HEADER_HEIGHT = 40
INTERNAL_HEIGHT = 600

DISPLAY_AI_WIDTH = 320      
DISPLAY_HUMAN_WIDTH = 1280  

SCREEN_WIDTH = DISPLAY_AI_WIDTH + DISPLAY_HUMAN_WIDTH  
SCREEN_HEIGHT = INTERNAL_HEIGHT + HEADER_HEIGHT         
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


class NumPyDQN:
    def __init__(self):
        try:
            self.w0 = np.load("net_0_weight.npy")
            self.b0 = np.load("net_0_bias.npy")
            self.w2 = np.load("net_2_weight.npy")
            self.b2 = np.load("net_2_bias.npy")
            self.w4 = np.load("net_4_weight.npy")
            self.b4 = np.load("net_4_bias.npy")
        except Exception:
            self.w0 = np.random.randn(64, 13).astype(np.float32) * 0.1
            self.b0 = np.zeros((64,), dtype=np.float32)
            self.w2 = np.random.randn(64, 64).astype(np.float32) * 0.1
            self.b2 = np.zeros((64,), dtype=np.float32)
            self.w4 = np.random.randn(4, 64).astype(np.float32) * 0.1
            self.b4 = np.zeros((4,), dtype=np.float32)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = self.relu(np.dot(self.w0, x) + self.b0)
        x = self.relu(np.dot(self.w2, x) + self.b2)
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


def draw_header(screen: pygame.Surface, font: pygame.font.Font) -> None:
    pygame.draw.rect(screen, GREY, (0, 0, SCREEN_WIDTH, HEADER_HEIGHT))

    ai_label = font.render("KI", True, CYAN)
    human_label = font.render("MENSCH", True, ORANGE)

    screen.blit(ai_label, ai_label.get_rect(center=(DISPLAY_AI_WIDTH // 2, HEADER_HEIGHT // 2)))
    screen.blit(human_label, human_label.get_rect(center=(DISPLAY_AI_WIDTH + (DISPLAY_HUMAN_WIDTH // 2), HEADER_HEIGHT // 2)))

    pygame.draw.line(screen, WHITE, (DISPLAY_AI_WIDTH, 0), (DISPLAY_AI_WIDTH, SCREEN_HEIGHT), 2)


async def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Lunar Lander - AI vs Human")

    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.SysFont("Arial", 20)
        header_font = pygame.font.SysFont("Arial", 24)
    except Exception:
        font = pygame.font.Font(None, 26)
        header_font = pygame.font.Font(None, 24)

    q_net = NumPyDQN()
    ai_state = GameState()
    human_state = GameState()
    reset_round(ai_state, human_state)
    last_meteor_id = id(ai_state.meteor)

    accumulator = 0.0
    ai_action = 0

    running = True
    while running:
        frame_time = clock.tick(FPS) / 1000.0
        frame_time = min(frame_time, MAX_FRAME_TIME)
        accumulator += frame_time

        round_over = is_round_over(ai_state) and is_round_over(human_state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    touch_x, touch_y = pygame.mouse.get_pos()
                    
                    if round_over:
                        box_width, box_height = 550, 100
                        box_x = (SCREEN_WIDTH - box_width) // 2
                        box_y = (SCREEN_HEIGHT - box_height) // 3
                        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
                        
                        hit_box = box_rect.inflate(40, 40)
                        
                        if hit_box.collidepoint(touch_x, touch_y):
                            reset_round(ai_state, human_state)
                            last_meteor_id = id(ai_state.meteor)
                            round_over = False

        keys = pygame.key.get_pressed()
        human_main_thrust = keys[pygame.K_UP] or keys[pygame.K_SPACE]
        human_rotate_left = keys[pygame.K_LEFT]
        human_rotate_right = keys[pygame.K_RIGHT]
        
        # Touch-/Maussteuerung für das getrennte Button-Layout im Human-Fenster
        if pygame.mouse.get_pressed()[0] and not round_over:
            touch_x, touch_y = pygame.mouse.get_pos()
            if touch_x > DISPLAY_AI_WIDTH and touch_y > HEADER_HEIGHT:
                local_x = touch_x - DISPLAY_AI_WIDTH
                local_y = touch_y - HEADER_HEIGHT
                
                btn_w, btn_h = 180, 70
                margin = 20
                spacing_y = 15
                
                bottom_y = INTERNAL_HEIGHT - btn_h - margin
                top_y = bottom_y - btn_h - spacing_y
                
                left_rect = pygame.Rect(margin, bottom_y, btn_w, btn_h)
                shub_left_rect = pygame.Rect(margin, top_y, btn_w, btn_h)
                right_rect = pygame.Rect(DISPLAY_HUMAN_WIDTH - btn_w - margin, bottom_y, btn_w, btn_h)
                shub_right_rect = pygame.Rect(DISPLAY_HUMAN_WIDTH - btn_w - margin, top_y, btn_w, btn_h)
                
                if left_rect.collidepoint(local_x, local_y):
                    human_rotate_left = True
                elif right_rect.collidepoint(local_x, local_y):
                    human_rotate_right = True
                elif shub_left_rect.collidepoint(local_x, local_y) or shub_right_rect.collidepoint(local_x, local_y):
                    human_main_thrust = True

        if keys[pygame.K_r] and round_over:
            reset_round(ai_state, human_state)
            last_meteor_id = id(ai_state.meteor)
            round_over = False

        if not is_round_over(ai_state):
            ai_action = get_ai_action(q_net, ai_state)

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

        while accumulator >= PHYSICS_DT:
            if not round_over:
                ai_state.update(PHYSICS_DT)
                human_state.update(PHYSICS_DT)
                last_meteor_id = sync_meteor_respawn(ai_state, human_state, last_meteor_id)
            accumulator -= PHYSICS_DT

        left_surface = pygame.Surface((DISPLAY_AI_WIDTH, INTERNAL_HEIGHT), pygame.SRCALPHA)
        right_surface = pygame.Surface((DISPLAY_HUMAN_WIDTH, INTERNAL_HEIGHT), pygame.SRCALPHA)

        draw_screen(left_surface, font, header_font, ai_state, current_inputs=None)
        
        human_inputs = {"left": human_rotate_left, "main": human_main_thrust, "right": human_rotate_right}
        draw_screen(right_surface, font, header_font, human_state, current_inputs=human_inputs)

        screen.fill(BLACK)
        screen.blit(left_surface, (0, HEADER_HEIGHT))
        screen.blit(right_surface, (DISPLAY_AI_WIDTH, HEADER_HEIGHT))
        
        draw_header(screen, header_font)

        if round_over:
            popup_bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            popup_bg.fill((0, 0, 0, 180))
            screen.blit(popup_bg, (0, 0))

            box_width, box_height = 550, 100
            box_rect = pygame.Rect(
                (SCREEN_WIDTH - box_width) // 2, 
                (SCREEN_HEIGHT - box_height) // 3, 
                box_width, 
                box_height
            )
            pygame.draw.rect(screen, GREY, box_rect, border_radius=10)
            pygame.draw.rect(screen, WHITE, box_rect, width=2, border_radius=10)

            line1 = font.render("Game Over!", True, WHITE)
            line2 = font.render("Drücke R (PC) oder tippe hier für eine neue Runde", True, CYAN)

            screen.blit(line1, line1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 )))
            screen.blit(line2, line2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 + 30)))

        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())