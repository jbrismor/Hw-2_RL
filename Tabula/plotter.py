import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

# Plot optimal policy for DP
class PolicyPlotter_DP:
    def __init__(self, grid_size, policy, walls=None, terminal_states=None, action_symbols=None):
        self.grid_size = grid_size
        # Ensure policy reshaping is correct
        self.policy = policy.reshape(grid_size)
        self.walls = set(walls) if walls is not None else set()
        self.terminal_states = set(terminal_states) if terminal_states is not None else set()
        
        # Define default or custom action symbols
        if action_symbols is None:
            self.action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        else:
            self.action_symbols = action_symbols

        # Validate that walls and terminal_states are provided as sets of tuples
        if any(not isinstance(cell, tuple) for cell in self.walls | self.terminal_states):
            raise ValueError("Each wall and terminal state must be a tuple (i, j)")

    # Plot the policy arrows on the grid world
    def plot(self):
        fig, ax = plt.subplots(figsize=(self.grid_size[1], self.grid_size[0]))
        ax.set_xticks(np.arange(self.grid_size[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.grid_size[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)

        # Plot the policy arrows while respecting walls and terminal states
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) in self.walls or (i, j) in self.terminal_states:
                    continue  # Skip walls and terminal states
                action = self.policy[i, j]
                arrow = self.action_symbols.get(action, '')
                ax.text(j, i, arrow, ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

        # Draw walls and terminal states with different colors
        for wall in self.walls:
            ax.add_patch(plt.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1, color='black'))
        for terminal in self.terminal_states:
            ax.add_patch(plt.Rectangle((terminal[1] - 0.5, terminal[0] - 0.5), 1, 1, color='red'))

        # Invert y-axis to match the row-column (i, j) convention
        plt.gca().invert_yaxis()
        plt.show()

# Plot the policy arrows for MC and TD
class PolicyPlotter_MC_TD:
    def __init__(self, grid_size, policy, walls=None, terminal_states=None, action_symbols=None):
        self.grid_size = grid_size
        self.policy = policy
        self.walls = set(walls) if walls is not None else set()
        self.terminal_states = set(terminal_states) if terminal_states is not None else set()
        
        # Define default or custom action symbols
        if action_symbols is None:
            self.action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        else:
            self.action_symbols = action_symbols

    # Plot the policy arrows on the grid world
    def plot(self):
        policy_array = np.array([self.policy.get(i, -1) for i in range(self.grid_size[0] * self.grid_size[1])])
        policy_array = policy_array.reshape(self.grid_size)

        fig, ax = plt.subplots(figsize=(self.grid_size[1], self.grid_size[0]))
        ax.set_xticks(np.arange(self.grid_size[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.grid_size[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) in self.walls:
                    # Draw wall cells in black
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))
                elif (i, j) in self.terminal_states:
                    # Draw terminal states in red
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red'))
                else:
                    action = policy_array[i, j]
                    arrow = self.action_symbols.get(action, '')
                    ax.text(j, i, arrow, ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

        # Invert y-axis to match the row-column (i, j) convention
        plt.gca().invert_yaxis()
        plt.show()

