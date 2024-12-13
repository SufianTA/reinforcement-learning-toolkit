# reinforcement-learning-toolkit/q_learning.py

import numpy as np

# Simple Q-learning algorithm
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((5, len(action_space)))  # Example state space with 5 states

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action])

# Example usage
agent = QLearningAgent(action_space=[0, 1, 2, 3])
state = 0
action = agent.choose_action(state)
agent.update_q_table(state, action, reward=1, next_state=1)
