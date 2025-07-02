# PongAI_DQN_MCTS

# 🏓 Pong AI vs Human

A classic Pong game where **a human plays against an AI opponent**, trained using a combination of **Deep Q-Learning (DQN)** and **Monte Carlo Tree Search (MCTS)**. The game features a graphical interface using `pygame` and machine learning powered by `PyTorch`.

---

## 🧠 Why Use DQN and MCTS?
This Pong AI project combines two powerful techniques — Deep Q-Network (DQN) and Monte Carlo Tree Search (MCTS) — to create a smarter, more strategic agent. Each method brings its own strengths, and together they form a well-balanced hybrid AI.
### 🎯 Why DQN?
Deep Q-Network (DQN) is a reinforcement learning algorithm that allows an agent to learn optimal actions through trial and error, using a neural network to estimate rewards.

Why DQN is used:
- Pong is a sequential decision-making game — perfect for reinforcement learning.
- DQN learns how to play by interacting with the environment and receiving feedback.
- It uses a neural network to estimate Q-values: how valuable a given action is in a given state.
- Over time, the AI improves its gameplay purely through self-play and experience.
Advantages of DQN:
- Scales well to complex environments (compared to traditional Q-tables).
- Learns optimal long-term strategies.
- Adapts to dynamic game states (e.g., ball speed, paddle movement).

### 🔍 Why MCTS?
Monte Carlo Tree Search (MCTS) is a planning algorithm that explores possible future outcomes through simulation before committing to an action.

Why MCTS is used:
- DQN learns good general behavior but lacks short-term foresight.
- MCTS performs lookahead planning — it simulates future states to anticipate better moves.
- It balances exploration (trying new paths) and exploitation (using the best-known options).
Advantages of MCTS:Provides short-term strategic depth by simulating future sequences.
- Improves decision-making in uncertain or high-pressure situations.
- Compensates for errors or limitations in early-stage DQN training.

### 🤝 Why Combine DQN + MCTS?
By combining both methods, we create an AI that can both learn from the past and think ahead:
| Component   | Role                                                |
| ----------- | --------------------------------------------------- |
| 🧠 **DQN**  | Learns general strategy from experience             |
| 🔍 **MCTS** | Performs real-time simulations to improve decisions |
This hybrid model allows the AI to:
- Learn over time from trial and error (DQN),
- Plan during gameplay by evaluating future possibilities (MCTS).

### 🔁 Simplified AI Workflow
During Training:
- DQN learns through repeated self-play.
During Gameplay:
- MCTS simulates future outcomes using Q-values as a guide.
- The action with the best outcome is chosen.
After Each Move:
- The AI stores the experience for future training updates.

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

---

## 💾 Model Saving & Loading  
To save a trained model:
 ```bash
torch.save(model.state_dict(), 'pong_dqn.pth')
```

To load a saved model:
 ```bash
model.load_state_dict(torch.load('pong_dqn.pth')
```

---
## 🔧 Configuration Tips

- You can modify pong_env.py to adjust game speed, paddle size, ball speed, or difficulty.
- For testing AI only: replace human paddle with scripted behavior.
- Use GPU (if available) to speed up training in train_ai.py.




  

  
