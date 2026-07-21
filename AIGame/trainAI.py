import os
import copy
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
EPISODES_TO_TRAIN = 4000 # Trainiere so viele Episoden

MODEL_PATH = "dataFromTraining.pth"

# Fine-Tuning: statt komplett neu zu trainieren, vorhandene Gewichte aus MODEL_PATH laden und mit
# geringerer Startexploration weitertrainieren - z.B. um nur das Landeverhalten zu verfeinern,
# ohne das bereits gelernte Ausweichverhalten zu verlernen.
FINE_TUNE_EPISODES = 1500
FINE_TUNE_EPSILON_START = 0.3

# DQN neigt dazu, eine einmal gute Policy im weiteren Training wieder zu "verlernen" (Oszillation).
# Statt nur die Gewichte der letzten Episode zu speichern, wird deshalb während des Trainings die
# beste je erreichte Landungsquote (Rolling-Fenster) getrackt und am Ende dieser Bestand gespeichert.
BEST_CHECKPOINT_WINDOW = 50

def rolling_rate(flags: list, window: int) -> np.ndarray:
    if len(flags) < window:
        return np.array([])
    return np.convolve(flags, np.ones(window) / window, mode='valid') * 100.0


def save_graph(episode_rewards: list, epsilon_values: list, outcomes: list):
    window = max(1, min(50, len(episode_rewards)))

    moving_avg = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode='valid'
    )

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 12))

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
    ax1.set_title("Lunar Lander DQN Training")

    # Episoden-Ausgang: Anteil (%) pro 50-Episoden-Fenster, getrennt nach Ursache.
    # bad_landing (über der Plattform, Schwellen verfehlt) und off_pad (komplett daneben gelandet)
    # getrennt geplottet, um zu erkennen, welcher der beiden Fehler tatsächlich überwiegt.
    landed = [1 if o == "landed" else 0 for o in outcomes]
    meteor_crash = [1 if o == "meteor" else 0 for o in outcomes]
    bad_landing = [1 if o == "bad_landing" else 0 for o in outcomes]
    off_pad = [1 if o == "off_pad" else 0 for o in outcomes]
    out_of_bounds = [1 if o == "out_of_bounds" else 0 for o in outcomes]

    x_range = range(window - 1, len(outcomes))
    ax3.plot(x_range, rolling_rate(landed, window), color="green", linewidth=2, label="Erfolgreiche Landung")
    ax3.plot(x_range, rolling_rate(meteor_crash, window), color="red", linewidth=2, label="Absturz durch Meteor")
    ax3.plot(x_range, rolling_rate(bad_landing, window), color="orange", linewidth=2, label="Absturz (Schwellen verfehlt)")
    ax3.plot(x_range, rolling_rate(off_pad, window), color="brown", linewidth=2, label="Absturz (neben Plattform)")
    ax3.plot(x_range, rolling_rate(out_of_bounds, window), color="purple", linewidth=2, label="Außerhalb der Welt")

    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Anteil (%), 50-Episoden-Fenster")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left")
    ax3.set_title("Episoden-Ausgang")

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=300, bbox_inches="tight")
    plt.close()

def train(episodes: int, epsilon_start: float, load_existing: bool):
    env = LunarLanderEnv(render_mode=False)

    # Netzwerke initialisieren (Online-Netz und stabiles Target-Netz)
    q_net = DQN(13, 4)
    target_net = DQN(13, 4)

    if load_existing:
        if os.path.exists(MODEL_PATH):
            q_net.load_state_dict(torch.load(MODEL_PATH))
            print(f"Fine-Tuning: Gewichte aus {MODEL_PATH} geladen.")
        else:
            print(f"Fine-Tuning: kein Modell unter {MODEL_PATH} gefunden - starte mit zufälligen Gewichten.")

    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Erfahrungsspeicher (Replay Buffer)
    replay_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = epsilon_start

    # Cache für den Graphen
    episode_rewards = []
    epsilon_values = []
    outcomes = []

    # Bestes je erreichtes Modell (nach Landungsquote im Rolling-Fenster) unabhängig vom Endstand
    best_landed_rate = -1.0
    best_state_dict = None


    for episode in range(episodes):
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
                
                # Target Q-Werte berechnen (Bellman-Gleichung, Double DQN):
                # Aktion wird vom Online-Netz gewählt, aber vom Target-Netz bewertet, um die
                # sonst übliche Überschätzung (beides vom selben Netz) zu vermeiden.
                with torch.no_grad():
                    best_actions = q_net(next_states_t).argmax(1, keepdim=True)
                    next_q = target_net(next_states_t).gather(1, best_actions)
                    target_q = rewards_t + (GAMMA * next_q * (1 - dones_t))
                
                # Gewichte updaten
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon verringern (weniger Zufall mit der Zeit)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        episode_rewards.append(episode_reward)
        epsilon_values.append(epsilon)

        # Episoden-Ausgang für die Erfolgs-/Absturz-Statistik festhalten
        lander = env.game_state.lander
        if lander.has_landed:
            outcomes.append("landed")
        else:
            outcomes.append(lander.death_reason or "unknown")

        # Bestes Modell festhalten: sobald die Landungsquote im Rolling-Fenster einen neuen
        # Bestwert erreicht, wird eine Kopie der aktuellen Gewichte gesichert. So geht die beste
        # Policy nicht verloren, falls das Training danach wieder schlechter wird.
        if len(outcomes) >= BEST_CHECKPOINT_WINDOW:
            recent_for_best = outcomes[-BEST_CHECKPOINT_WINDOW:]
            current_landed_rate = recent_for_best.count("landed") / BEST_CHECKPOINT_WINDOW * 100.0
            if current_landed_rate > best_landed_rate:
                best_landed_rate = current_landed_rate
                best_state_dict = copy.deepcopy(q_net.state_dict())
                print(f"Episode {episode} | Neuer Bestwert: Landung(50): {best_landed_rate:.0f}%")

        # Target-Netzwerk regelmäßig synchronisieren
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            recent = outcomes[-50:]
            landed_rate = recent.count("landed") / len(recent) * 100.0
            meteor_rate = recent.count("meteor") / len(recent) * 100.0
            bad_landing_rate = recent.count("bad_landing") / len(recent) * 100.0
            off_pad_rate = recent.count("off_pad") / len(recent) * 100.0
            out_of_bounds_rate = recent.count("out_of_bounds") / len(recent) * 100.0
            print(
                f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.2f} | "
                f"Landung(50): {landed_rate:.0f}% | Meteor-Absturz(50): {meteor_rate:.0f}% | "
                f"Schwellen-Absturz(50): {bad_landing_rate:.0f}% | Neben-Plattform(50): {off_pad_rate:.0f}% | "
                f"Außerhalb-Welt(50): {out_of_bounds_rate:.0f}%"
            )

    # Bestes je erreichtes Modell speichern (falls das Rolling-Fenster nie erreicht wurde,
    # z.B. bei sehr kurzen Testläufen, wird ersatzweise der aktuelle Stand gespeichert)
    if best_state_dict is not None:
        torch.save(best_state_dict, MODEL_PATH)
        print(f"Bestes Modell gespeichert (Landung(50): {best_landed_rate:.0f}%).")
    else:
        torch.save(q_net.state_dict(), MODEL_PATH)
        print("Kein Best-Checkpoint erreicht - aktueller Stand gespeichert.")

    save_graph(episode_rewards, epsilon_values, outcomes)
    print("Training beendet. Modell gespeichert!")


def fine_tune(episodes: int = FINE_TUNE_EPISODES, epsilon_start: float = FINE_TUNE_EPSILON_START):
    """
    Lädt die vorhandenen Gewichte aus MODEL_PATH und trainiert mit geringerer Startexploration
    weiter, statt komplett neu zu beginnen - z.B. um nur das Landeverhalten zu verfeinern, ohne
    bereits gelerntes Ausweichverhalten zu verlernen.
    """
    train(episodes=episodes, epsilon_start=epsilon_start, load_existing=True)


def main():
    """
    Einheitlicher Einstiegspunkt: existiert bereits ein Modell unter MODEL_PATH, wird darauf per
    Fine-Tuning weitertrainiert. Andernfalls wird ein komplett neues Modell von Null trainiert.
    """
    if os.path.exists(MODEL_PATH):
        fine_tune()
    else:
        train(episodes=EPISODES_TO_TRAIN, epsilon_start=EPSILON_START, load_existing=False)


if __name__ == "__main__":
    main()