import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class Solver:
    def __init__(self, env, gamma=0.99, theta=0.01, use_value_iteration=False):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = np.zeros(env.observation_space.n, dtype=int)
        self.use_value_iteration = use_value_iteration
        self.V_history = []
        self.Q_history = []
        self.policy_history = []

    def value_iteration(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                # Calculate action values and update V[s] based on these values
                action_values = [self.expected_return(s, a) for a in range(self.env.action_space.n)]
                max_action_value = max(action_values)
                self.V[s] = max_action_value
                delta = max(delta, abs(v - self.V[s]))
                
                # Check if policy change is needed
                new_policy = np.argmax(action_values)
                if new_policy != self.policy[s]:
                    self.policy[s] = new_policy  # Update policy at state s if there is a change
                
            self.V_history.append(self.V.copy())
            self.Q_history.append(self.Q.copy())
            self.policy_history.append(self.policy.copy())  # Record policy after each iteration
        
            if delta < self.theta:
                break

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                self.V[s] = max(self.expected_return(s, a) for a in range(self.env.action_space.n))
                delta = max(delta, abs(v - self.V[s]))
                self.V_history.append(self.V.copy())
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.env.observation_space.n):
            old_action = self.policy[s]
            action_values = [self.expected_return(s, a) for a in range(self.env.action_space.n)]
            self.Q[s] = action_values
            self.policy[s] = np.argmax(action_values)
            self.Q_history.append(self.Q.copy())
            if old_action != self.policy[s]:
                policy_stable = False
        self.policy_history.append(self.policy.copy())
        return policy_stable

    def expected_return(self, state, action):
        return sum(prob * (reward + self.gamma * self.V[next_state]) for prob, next_state, reward in self.env.get_transitions(state, action))

    def plot_policy(self):
        plt.figure(figsize=(8, 4))
        ax = plt.gca()  # Get current axis

        policy_arrows = ['←', '→']  # Only two actions: left and right
        policy_labels = [policy_arrows[action] for action in self.policy]

        # Create grid layout
        ax.set_xlim(0, len(policy_labels))
        ax.set_ylim(0, 1)
        ax.set_xticks([0.5 + i for i in range(len(policy_labels))])  # Positioning x-ticks in the middle of the cells
        ax.set_xticklabels(['State ' + str(i) for i in range(len(policy_labels))])  # Labeling states
        ax.set_yticks([])  # No y-ticks needed

        # Add grid lines
        for i in range(len(policy_labels) + 1):
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, fill=None, edgecolor='black', lw=2))

        # Plot arrows in the middle of each grid cell
        for i, label in enumerate(policy_labels):
            ax.text(i + 0.5, 0.5, label, ha='center', va='center', fontsize=30, fontweight='bold')

        plt.title("Optimal Policy")
        plt.xlabel('State')
        plt.show()

    def plot_convergence(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.plot([np.mean(v) for v in self.V_history])
        plt.title('Convergence of State-Value Function')
        plt.xlabel('Iteration')
        plt.ylabel('Average State-Value')

        plt.subplot(1, 3, 2)
        plt.plot([np.mean(q) for q in self.Q_history])
        plt.title('Convergence of Action-Value Function')
        plt.xlabel('Iteration')
        plt.ylabel('Average Action-Value')

        plt.subplot(1, 3, 3)
        policy_changes = [np.sum(p != self.policy_history[i - 1]) if i > 0 else self.env.observation_space.n for i, p in enumerate(self.policy_history)]
        plt.plot(policy_changes)
        plt.title('Policy Changes per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Changed Actions')

        plt.tight_layout()
        plt.show()

    def solve(self):
        if self.use_value_iteration:
            self.value_iteration()
        else:
            while True:
                self.policy_evaluation()
                if self.policy_improvement():
                    break
        self.plot_policy()
        self.plot_convergence()
        return self.policy, self.V, self.Q