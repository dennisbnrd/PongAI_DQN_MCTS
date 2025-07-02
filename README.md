# PongAI_DQN_MCTS

# 🏓 Pong AI vs Human

A classic Pong game where **a human plays against an AI opponent**, trained using a combination of **Deep Q-Learning (DQN)** and **Monte Carlo Tree Search (MCTS)**. The game features a graphical interface using `pygame` and machine learning powered by `PyTorch`.

---

## 🎮 Features

- Game mode: **Human vs AI**
- Pong court designed like a real table tennis field
- Automatic scoring system
- AI learns using Reinforcement Learning (DQN) and MCTS decision-making
- Save and load AI model easily
- Offline training and real-time live training support

---

## 🧠 Technologies Used

- Python 3.10+
- [PyTorch](https://pytorch.org/) — for building and training the AI model
- [Pygame](https://www.pygame.org/) — for graphical user interface
- AI Algorithms:
  - Deep Q-Network (DQN)
  - Monte Carlo Tree Search (MCTS)

---

## 📁 Project Structure

pong-ai-vs-human/

- pong_env.py       # Game environment (rules, ball/paddle movement)
- dqn_model.py        # DQN model definition
- mcts_agent.py       # MCTS agent for decision making
- train_ai.py         # AI training script
- play_Ai_manusia.py  # Game script: play vs AI
- pong_dqn.pth        # Trained model file 
- README.md           # This file
- requirements.txt    # Python dependencies

---
## ⚙️ Instalasi
1. Clone Repositori:
```bash
git clone https://github.com/dennisbnrd/PongAI_DQN_MCTS.git
cd PongAI_DQN_MCTS-CRUD
```
2. Install dependencies:
  ```bash
pip install -r requirements.txt
```

---
## 🚀 How to Run
1. ▶️ Play the game (Human vs AI)
  ```bash
python play_ai.py
```
2. 🧠 Train the AI
 ```bash
python train_ai.py
```  
This will train the DQN model and save it as pong_dqn.pth.

## 💾 Model Saving & Loading  
  

  
