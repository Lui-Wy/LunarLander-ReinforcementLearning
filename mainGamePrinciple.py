import pygame
import sys
import math
import random

# --- KONSTANTEN ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Physik (Werte sind für ein gutes Spielgefühl angepasst)
GRAVITY = 0.05
THRUST = 0.15
ROTATE_SPEED = 3  # in Grad pro Frame

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class Lander:
    def __init__(self, x, y):
        # Position und Bewegung
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0  # 0 Grad bedeutet: Nase zeigt nach oben
        
        # Zustand
        self.fuel = 500
        self.is_alive = True
        self.has_landed = False
        
        # Triebwerk-Animation Flags
        self.main_thrust_on = False
        self.left_thrust_on = False
        self.right_thrust_on = False

    def update(self):
        if not self.is_alive or self.has_landed:
            return

        # 1. Schwerkraft wirkt immer nach unten
        self.vy += GRAVITY

        # 2. Winkel in Bogenmaß für Trigonometrie umrechnen
        # Pygame dreht im Uhrzeigersinn, mathematisch ist gegen den Uhrzeigersinn. 
        # Da 0 Grad "oben" sein soll, passen wir die Achsen an.
        rad = math.radians(self.angle)

        # 3. Triebwerke verarbeiten
        if self.main_thrust_on and self.fuel > 0:
            self.vx += THRUST * math.sin(rad)
            self.vy -= THRUST * math.cos(rad)
            self.fuel -= 1

        if self.left_thrust_on and self.fuel > 0:
            self.angle += ROTATE_SPEED
            self.fuel -= 0.2

        if self.right_thrust_on and self.fuel > 0:
            self.angle -= ROTATE_SPEED
            self.fuel -= 0.2

        # 4. Position updaten
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface):
        # Wir zeichnen ein einfaches Dreieck für die Landefähre
        # Um es zu rotieren, berechnen wir die Punkte relativ zum Zentrum
        points = [
            (0, -15),  # Spitze
            (-10, 15), # Links unten
            (10, 15)   # Rechts unten
        ]
        
        rotated_points = []
        rad = math.radians(self.angle)
        
        for px, py in points:
            # Rotationsmatrix anwenden
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            rotated_points.append((self.x + rx, self.y + ry))
            
        # Farbe je nach Zustand
        color = GREEN if self.has_landed else (RED if not self.is_alive else WHITE)
        pygame.draw.polygon(surface, color, rotated_points, 2)
        
        # Triebwerksflamme zeichnen
        if self.main_thrust_on and self.fuel > 0 and self.is_alive and not self.has_landed:
            flame_points = [(-5, 16), (0, 28), (5, 16)]
            rotated_flame = []
            for fx, fy in flame_points:
                rx = fx * math.cos(rad) - fy * math.sin(rad)
                ry = fx * math.sin(rad) + fy * math.cos(rad)
                rotated_flame.append((self.x + rx, self.y + ry))
            pygame.draw.polygon(surface, YELLOW, rotated_flame)


def main():
    pygame.init()
    screen = pygame.display.set_index = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lunar Lander - Phase 1 (Menschliches Spiel)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Lander initialisieren (Mitte des Bildschirms, oben)
    lander = Lander(SCREEN_WIDTH // 2, 100)
    
    # Einfache Landeplattform definieren (X-Start, X-Ende, Y-Höhe)
    pad_x_start = 350
    pad_x_end = 450
    pad_y = 550

    running = True
    while running:
        clock.tick(FPS)
        screen.fill(BLACK)

        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Tastendrücke abfragen (kontinuierlich)
        keys = pygame.key.get_pressed()
        if lander.is_alive and not lander.has_landed:
            lander.main_thrust_on = keys[pygame.K_UP] or keys[pygame.K_SPACE]
            lander.left_thrust_on = keys[pygame.K_RIGHT]  # Dreht nach rechts
            lander.right_thrust_on = keys[pygame.K_LEFT]   # Dreht nach links
        else:
            lander.main_thrust_on = False
            lander.left_thrust_on = False
            lander.right_thrust_on = False

        # --- UPDATE ---
        lander.update()

        # --- KOLLISIONSABFRAGE (LOGIK) ---
        # 1. Check ob Plattform getroffen wurde
        if lander.y >= pad_y - 15:  # 15 ist der halbe Lander-Radius
            if pad_x_start <= lander.x <= pad_x_end:
                # Landebedingungen prüfen: Sanfte Geschwindigkeit & aufrechter Winkel
                if abs(lander.vy) < 1.5 and abs(lander.vx) < 1.0 and abs(lander.angle % 360) < 10:
                    lander.has_landed = True
                    lander.vy = 0
                    lander.vx = 0
                else:
                    lander.is_alive = False
            else:
                # Neben der Plattform aufgekommen -> Crash
                lander.is_alive = False
                
        # 2. Check ob aus dem Bildschirm geflogen
        if lander.x < 0 or lander.x > SCREEN_WIDTH or lander.y < 0:
            lander.is_alive = False

        # --- DRAW ---
        # Landeplattform zeichnen
        pygame.draw.line(screen, GREEN, (pad_x_start, pad_y), (pad_x_end, pad_y), 5)
        
        # Lander zeichnen
        lander.draw(screen)

        # UI / Telemetrie-Daten anzeigen
        color_vx = GREEN if abs(lander.vx) < 1.0 else RED
        color_vy = GREEN if abs(lander.vy) < 1.5 else RED
        
        ui_fuel = font.render(f"Treibstoff: {int(lander.fuel)}", True, WHITE)
        ui_vx = font.render(f"H-Geschw.: {lander.vx:.2f}", True, color_vx)
        ui_vy = font.render(f"V-Geschw.: {lander.vy:.2f}", True, color_vy)
        ui_angle = font.render(f"Winkel: {int(lander.angle)}°", True, WHITE)
        
        screen.blit(ui_fuel, (10, 10))
        screen.blit(ui_vx, (10, 35))
        screen.blit(ui_vy, (10, 60))
        screen.blit(ui_angle, (10, 85))

        # Spielende-Text
        if not lander.is_alive:
            txt = font.render("CRASH! Drücke R für Reset", True, RED)
            screen.blit(txt, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))
            if keys[pygame.K_r]: lander = Lander(SCREEN_WIDTH // 2, 100)
        elif lander.has_landed:
            txt = font.render("ERFOLGREICHE LANDUNG! Drücke R für Reset", True, GREEN)
            screen.blit(txt, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
            if keys[pygame.K_r]: lander = Lander(SCREEN_WIDTH // 2, 100)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()