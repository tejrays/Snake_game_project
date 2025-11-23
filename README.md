# Snake AI — Q-Learning + Deep Q-Learning (DQN)

A complete implementation of Reinforcement Learning for Snake, featuring:

* Tabular Q-Learning
* Deep Q-Learning (DQN with Replay Buffer)
* Human-playable Snake
* Live training graphs
* Side-by-side comparison

---

## Project Structure

```
├── game.py                   # Snake environment for AI
├── snake_game_human.py       # Playable Snake (keyboard)
├── snake_env_discrete.py     # Discrete-state environment for Q-Learning
├── qlearning.py              # Tabular Q-Learning agent
├── model.py                  # Neural network + trainer (DQN)
├── agent.py                  # Deep Q-Learning agent (training loop)
├── helper.py                 # Live graph plotting utility
├── compare.py                # Runs Q-Learning → DQN sequentially
└── README.md
```

---

## Installation

### 1. Install Required Packages

```bash
pip install pygame torch numpy matplotlib
```

### 2. Make Sure `arial.ttf` Is Present

Place **arial.ttf** in the same folder.

---

## Run the Human Snake Game

Play the game using arrow keys.

```bash
python snake_game_human.py
```

---

## Train Using Tabular Q-Learning

The discrete environment + Q-table agent learns basic survival.

### Run Q-Learning:

```bash
python compare.py
```

This will:

* Train Q-learning
* Show live score graph
* Print episode results

---

## Train Using Deep Q-Learning (DQN)

DQN uses:

* Neural Network (11 → 256 → 3)
* Experience Replay
* ε-greedy exploration
* Batch training

### Run DQN:

```bash
python agent.py
```

This will:

* Train the neural network
* Save best model
* Show live graph

---

## Compare Q-Learning & DQN

Run both back-to-back:

```bash
python compare.py
```

This script:

* Trains Q-Learning
* Starts DQN next
* Shows training graphs

