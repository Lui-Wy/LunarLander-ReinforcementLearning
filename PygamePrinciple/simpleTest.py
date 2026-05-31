import pygame

pygame.init()

##Game Window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

##Game Loop
running = True
while running:
    #Event Handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()