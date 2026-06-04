import torch
import pygame
from AIGame.codeForAI import LunarLanderEnv
from AIGame.trainAI import DQN

def main():
    env = LunarLanderEnv(render_mode=True)
    q_net = DQN(6, 4)
    q_net.load_state_dict(torch.load("dataFromTraining_1.pth"))
    q_net.eval() # Netzwerk in Testmodus versetzen

    for episode in range(5):
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
                
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = q_net(state_t).argmax().item()
                
            state, reward, done = env.step(action)
            env.render()

if __name__ == "__main__":
    main()