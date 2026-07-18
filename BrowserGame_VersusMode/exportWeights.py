import torch
import numpy as np
# Importiere hier deine originale DQN Klasse
from trainAI import DQN

# 1. Modell instanziieren und die gelernten Gewichte laden
# (Dein state_dim ist 13 wegen der Meteoriten, action_dim ist 4)
model = DQN(state_dim=13, action_dim=4)
model.load_state_dict(torch.load("dataFromTraining.pth", map_location="cpu"))
model.eval()

# 2. Gewichte gezielt aus der Sequenz extrahieren
state_dict = model.state_dict()

# Layer 0 (Erster Linear-Layer)
np.save("net_0_weight.npy", state_dict["net.0.weight"].numpy())
np.save("net_0_bias.npy", state_dict["net.0.bias"].numpy())

# Layer 2 (Zweiter Linear-Layer)
np.save("net_2_weight.npy", state_dict["net.2.weight"].numpy())
np.save("net_2_bias.npy", state_dict["net.2.bias"].numpy())

# Layer 4 (Output Linear-Layer)
np.save("net_4_weight.npy", state_dict["net.4.weight"].numpy())
np.save("net_4_bias.npy", state_dict["net.4.bias"].numpy())

print("Super! Alle Gewichte wurden passend zu deiner Sequential-Struktur als .npy exportiert.")