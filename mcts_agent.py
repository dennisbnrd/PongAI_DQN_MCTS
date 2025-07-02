import torch
import copy
import numpy as np

class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visits = 0
        self.value = 0.0

class MCTSAgent:
    def __init__(self, model, simulations=100):
        self.model = model
        self.simulations = simulations

    def select_action(self, env, state):
        root = MCTSNode(state)

        for _ in range(self.simulations):
            self.simulate(env, root)

        best_action = max(root.children.items(), key=lambda item: item[1].value / (item[1].visits + 1e-6))[0]
        return best_action

    def simulate(self, env, node):
        if node.visits == 0:
            with torch.no_grad():
                value = self.model(torch.tensor(node.state, dtype=torch.float32)).max().item()
            node.value += value
            node.visits += 1
            return value

        if not node.children:
            for action in [0, 1, 2]:
                env_copy = copy.deepcopy(env)
                next_state, reward, done, _ = env_copy.step(action)
                if done:
                    value = reward
                else:
                    with torch.no_grad():
                        value = reward + self.model(torch.tensor(next_state, dtype=torch.float32)).max().item()
                child = MCTSNode(next_state)
                child.value = value
                child.visits = 1
                node.children[action] = child
            return value

        best_child = max(node.children.items(), key=lambda item: item[1].value / (item[1].visits + 1e-6))[1]
        value = self.simulate(env, best_child)
        best_child.value += value
        best_child.visits += 1
        return value
