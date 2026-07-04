import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from AIGame.codeForAI import LunarLanderEnv

# --- NEURONALES NETZ (Das Gehirn) ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Ein einfaches Netz mit 2 verdeckten Schichten (6 Inputs -> 64 -> 64 -> 4 Outputs)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- HYPERPARAMETER ---
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
EPISODES_TO_TRAIN = 1000 # Trainiere so viele Episoden

def save_graph(episode_rewards: list, epsilon_values: list):
    window = 50

    moving_avg = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode='valid'
    )

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Rohdaten
    ax1.plot(
        episode_rewards,
        alpha=0.2,
        label="Reward"
    )

    # Durchschnitt
    ax1.plot(
        range(window - 1, len(episode_rewards)),
        moving_avg,
        linewidth=2,
        label="50 Episoden Durchschnitt"
    )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Epsilon rechts
    ax2 = ax1.twinx()
    ax2.plot(
        epsilon_values,
        linestyle="--",
        linewidth=2,
        label="Epsilon"
    )
    ax2.set_ylabel("Epsilon")

    # Gemeinsame Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower left",
    )

    plt.title("Lunar Lander DQN Training")
    plt.savefig("training_progress.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    env = LunarLanderEnv(render_mode=False)
    
    # Netzwerke initialisieren (Online-Netz und stabiles Target-Netz)
    q_net = DQN(13, 4)
    target_net = DQN(13, 4)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    # Erfahrungsspeicher (Replay Buffer)
    replay_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    # Cache für den Graphen
    episode_rewards = []
    epsilon_values = []

    
    for episode in range(EPISODES_TO_TRAIN):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy Strategie (Zufall vs. Netzwissen)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = q_net(state_t).argmax().item()
            
            # Schritt in der Umgebung machen
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Im Speicher ablegen
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            # Wenn genug Erfahrungen gesammelt wurden -> Trainieren!
            if len(replay_buffer) > MIN_REPLAY_SIZE:
                # Batch zufällig auswählen
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones).unsqueeze(1)
                
                # Aktuelle Q-Werte berechnen
                current_q = q_net(states_t).gather(1, actions_t)
                
                # Target Q-Werte berechnen (Bellman-Gleichung)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))
                
                # Gewichte updaten
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Epsilon verringern (weniger Zufall mit der Zeit)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        episode_rewards.append(episode_reward)
        epsilon_values.append(epsilon)

        # Target-Netzwerk regelmäßig synchronisieren
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.2f}")

    # Modell manuell speichern
    torch.save(q_net.state_dict(), "dataFromTraining.pth")
    save_graph(episode_rewards, epsilon_values)
    print("Training beendet. Modell gespeichert!")

if __name__ == "__main__":
    main()