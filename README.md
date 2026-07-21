LunarLander-ReinforcementLearning/
│
├── core/
│   ├── __init__.py
│   ├── game_logic.py
│   └── rendering.py
│
├── HumanGame/
│   ├── __init__.py
│   └── main.py
│
├── AIGame/
│   ├── __init__.py
│   ├── codeForAI.py
│   ├── trainAI.py
│   └── realTestOfAI.py
│
├── VersusGame/
│   ├── __init__.py
│   └── main.py
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
    "pip install pygame numpy torch matplotlib"
    oder 
    "pip install pygame-ce numpy torch matplotlib"
    installiert werden

VORAB: Alle Befehle werden aus dem Projektverzeichnis ausgeführt.

Ziel des Spiels: sicher auf der grünen Plattform landen (langsam, gerade, zentriert), ohne
mit dem umherfliegenden Meteor zu kollidieren oder aus dem Bildschirm hinauszufliegen.

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

    Existiert bereits "dataFromTraining.pth", wird darauf per Fine-Tuning weitertrainiert
    (geringere Startexploration), statt komplett neu zu beginnen. Existiert es nicht, startet
    ein komplett neues Training. Gespeichert wird am Ende jeweils das beste je erreichte Modell
    (nach Landungsquote), nicht zwangsläufig der allerletzte Trainingsstand.
    !!!Sichere also ältere Trainingsmodelle unter einem anderen Namen, sonst werden diese überschrieben!!!

    Zusätzlich wird "training_progress.png" gespeichert: Reward/Epsilon-Verlauf sowie eine
    Aufschlüsselung der Episoden-Ausgänge (Landung, Meteor-Absturz, Schwellen verfehlt,
    neben der Plattform, außerhalb der Welt).

Trainierte KI testen:
    "python -m AIGame.realTestOfAI"
    
    verwendet das aktuell unter dem Namen "dataFromTraining.pth" gespeicherte Modell für den Test. 

Versus-Modus (KI gegen Mensch, Splitscreen):
    "python -m VersusGame.main"

    verwendet das aktuell unter dem Namen "dataFromTraining.pth" gespeicherte Modell für die KI-Seite.
    Beide Seiten starten in jeder Runde mit identischer Startposition, Plattformposition und Meteor.

    Steuerung (menschliche Seite, rechts):
    | Taste              | Aktion                      |
    | ------------------ | --------------------------- |
    | Pfeil hoch / SPACE | Haupttriebwerk              |
    | Pfeil links        | Nach links rotieren         |
    | Pfeil rechts       | Nach rechts rotieren        |
    | R                  | Neue Runde, sobald beide Seiten gelandet oder gecrasht sind |