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
<pre>
pong-ai-vs-human/
│
├── pong_env.py         # Game environment (rules, ball/paddle movement)
├── dqn_model.py        # DQN model definition
├── mcts_agent.py       # MCTS agent for decision making
├── train_ai.py         # AI training script
├── play_ai.py          # Game script: play vs AI
├── utils.py            # Helper functions (save/load)
├── pong_dqn.pth        # Trained model file (optional)
├── README.md           # This file
└── requirements.txt    # Python dependencies
<pre>
