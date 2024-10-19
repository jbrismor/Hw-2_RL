import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd

# Boat example (code from website)
class BoatEnv(gym.Env):

    metadata = {'render_mods': [None]}

    def __init__(self, east_wind=0.7, west_wind=0.3, seed=None):
        """
        Arguments:
            east_wind (float) : Probability of easterly wind
        """
        
        assert east_wind + west_wind <= 1, 'Invalid wind probabilities'
        
        # Only 2 states "left state or right state"
        self.observation_space = spaces.Discrete(2)

        # Two possible actions motor on or motor off
        self.action_space = spaces.Discrete(2)

        # Define a random number generator
        self.rng = np.random.default_rng(seed)

        # Probabilities of wind
        self.prob_wind = [east_wind, west_wind, 1-east_wind-west_wind]

    def get_info(self, wind):
        
        direction = {0: 'No Wind', 1: 'East Wind', -1:'West Wind'}

        if wind is not None:
            info = {'Wind': direction[wind]}
        else:
            info = {'Other News': 'Nothing to report Huston'}

        return info

    def reset(self):
        
        # Always start in the left state
        self.state = 0
        
        observation = self.state
        info = self.get_info(None)

        return observation, info
    
    def step(self, action):

        # East +1, No Wind 0, West -1, note wind blows boat plus 1 minus 1 or 
        # does not move boat at all
        wind = self.rng.choice([1, 0, -1], p=self.prob_wind)

        # Determine reward (0, 1, 2, 3, or 4)
        reward = 2*self.state+1 + wind + action

        # Update the state (s')
        if self.state == 0: 
            if reward<2:
                self.state = 0
            else:
                self.state = 1
        else:
            if reward>2:
                self.state = 1
            else:
                self.state = 0

        observation = self.state
        terminated = False
        truncated = False
        info = self.get_info(wind)

        return observation, reward, terminated, truncated, info

  
# Grid World
class GridWorldEnv(gym.Env):
    def __init__(self, gamma=0.25):
        super(GridWorldEnv, self).__init__()
        
        # Grid dimensions
        self.grid_size = (6, 6)
        
        # Actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        
        # Observations: (i, j) coordinates in the grid
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size[0]),
            spaces.Discrete(self.grid_size[1])
        ))
        
        # Define the grid structure
        self.walls = [(i, 2) for i in [0, 1]] + [(i, 2) for i in [3, 4, 5]] + [(3, j) for j in [2, 3, 4]]  # Blue walls (bounce off)
        # terminal states
        self.terminal_states = {
            (5, 0): -50,  # Red state
            (1, 4): -50,  # Red state
            (5, 5): 100   # Green state
        }
        # step penalty
        self.default_reward = -1  # Penalty for each step
        
        # Movement noise (probability that the action fails and a random action occurs)
        self.gamma = gamma
        self.rng = np.random.default_rng()
        
        # Start the agent in the (0, 0) position
        self.state = (0, 0)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        # Introduce noise: the robot might not take the desired action
        if self.rng.random() < self.gamma:
            action = self.rng.choice([0, 1, 2, 3])  # Random action
        
        i, j = self.state
        
        # Move according to action: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        if action == 0:  # Up
            next_state = (max(i - 1, 0), j)
        elif action == 1:  # Right
            next_state = (i, min(j + 1, self.grid_size[1] - 1))
        elif action == 2:  # Down
            next_state = (min(i + 1, self.grid_size[0] - 1), j)
        elif action == 3:  # Left
            next_state = (i, max(j - 1, 0))
        
        # Bounce off walls
        if next_state in self.walls:
            next_state = self.state  # Stay in the same state
        
        # Assign rewards and check if the agent is in a terminal state
        if next_state in self.terminal_states:
            reward = self.terminal_states[next_state]
            done = True
        else:
            reward = self.default_reward
            done = False
        
        # Update state
        self.state = next_state
        info = {}
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        # This is where rendering (like printing the grid) could happen
        grid = np.full(self.grid_size, ' ')
        grid[self.state] = 'A'  # Mark the agent's position
        for wall in self.walls:
            grid[wall] = 'W'  # Mark walls
        for terminal in self.terminal_states:
            if self.terminal_states[terminal] > 0:
                grid[terminal] = 'G'  # Green for the goal
            else:
                grid[terminal] = 'R'  # Red for hazards
        
        print("\n".join(["".join(row) for row in grid]))