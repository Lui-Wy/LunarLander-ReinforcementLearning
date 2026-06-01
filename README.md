LunarLander-ReinforcementLearning/
│
├── game_logic.py
│
├── HumanGame/
│   ├── __init__.py
│   ├── main.py
│   └── rendering.py
│
├── AIGame/
│   ├── __init__.py
│   ├── codeForAI.py
│   ├── trainAI.py
│   └── realTestOfAI.py
│
├── requirements.txt
└── README.md

Vorraussetzungen:
Python 3.12+

Installation:
- gehe in das Projektverzeichnis per Terminal:
    "cd [Pfad zum Projektverzeichnis]/LunarLander-ReinforcementLearning"
- erstelle eine virtuelle Umgebung, um unabhängig lokalen Rechner zu sein
    "python -m venv .venv"
- aktiviere die neu erstellte virtuelle Umgebung (wenn man Windiws CMD verwendet activate.bat anstatt Activate.ps1)
    ".venv\Scripts\Activate.ps1"
- Abhängigkeiten installieren (wenn die verwendete Python-Version < 3.14 sit tausche "pygame-ce" durch "pygame" in der requirements.txt)
    "pip install -r requirements.txt"

    Alternativ können di Packages manuell mit 
    "pip install pygame numpy stable-baselines3 torch"
    oder 
    "pip install pygame-ce numpy stable-baselines3 torch"
    installiert werden

VORAB: Alle Befehle werden aus dem Projektverzeichnis ausgeführt.

Spiel für den Menschen starten:
    "python -m HumanGame.main"

    Steuerung:
    | Taste              | Aktion                      |
    | ------------------ | --------------------------- |
    | Pfeil hoch / SPACE | Haupttriebwerk              |
    | Pfeil links        | Nach links rotieren         |
    | Pfeil rechts       | Nach rechts rotieren        |
    | R                  | Neustart nach Crash/Landung |

KI trainieren:
    "python -m AIGame.trainAI"

    Das Modell wird nach abschluss des Trainings in "dataFromTraining.pth" gespeichert. 
    !!!Sichere also ältere Trainingsmodelle unter einem anderen Namen, sonst werden diese überschrieben!!!

    Zusätzlich Lernerfolg beim Training (Reward) und der Zufallswert als "training_progress.png" abgespeichert.

Trainierte KI testen:
    "python -m AIGame.realTestOfAI"
    
    verwendet das aktuell unter dem Namen "dataFromTraining.pth" gespeicherte Modell für den Test. 