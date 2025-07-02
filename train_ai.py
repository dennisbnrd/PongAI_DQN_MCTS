import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from pong_env import PongEnv
from dqn_model import DQN

env = PongEnv()
model = DQN(input_dim=6, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
replay_buffer = deque(maxlen=10000)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = model(state_tensor)
        return q_values.argmax().item()

def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = model(next_states).max(1)[0]
    target_q = rewards + 0.99 * next_q_values * (1 - dones)
    loss = F.mse_loss(q_values, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop
epsilon = 1.0
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= 64:
            batch = random.sample(replay_buffer, 64)
            train_step(batch)

    epsilon *= 0.995
    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# Save model
torch.save(model.state_dict(), "pong_dqn.pth")
print("✅ Model berhasil disimpan ke pong_dqn.pth")