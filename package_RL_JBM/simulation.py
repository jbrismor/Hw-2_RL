import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
import pygame

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

        # Pygame setup
        pygame.init()
        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Boat Environment")
        self.clock = pygame.time.Clock()

        # Load boat image or use a rectangle for the boat
        self.boat_image = pygame.image.load('package_RL_JBM/boat.jpg')
        self.boat_image = pygame.transform.scale(self.boat_image, (200, 100))

        # Colors
        self.bg_color = (173, 216, 230)  # Light blue background (water)
        self.line_color = (0, 0, 0)  # Black lines
        self.end_line_color = (255, 165, 0)  # Orange lines

    def get_info(self, wind):
        direction = {0: 'No Wind', 1: 'East Wind', -1: 'West Wind'}

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
        # East +1, No Wind 0, West -1
        wind = self.rng.choice([1, 0, -1], p=self.prob_wind)

        # Determine reward (0, 1, 2, 3, or 4)
        reward = 2*self.state + 1 + wind + action

        # Update the state (s')
        if self.state == 0: 
            if reward < 2:
                self.state = 0
            else:
                self.state = 1
        else:
            if reward > 2:
                self.state = 1
            else:
                self.state = 0

        observation = self.state
        terminated = False
        truncated = False
        info = self.get_info(wind)

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Clear the screen with background color
        self.screen.fill(self.bg_color)

        # Draw vertical lines: orange for ends, black for middle
        pygame.draw.line(self.screen, self.end_line_color, (0, 0), (0, self.screen_height), 10)  # Left orange line
        pygame.draw.line(self.screen, self.end_line_color, (self.screen_width - 10, 0), (self.screen_width - 10, self.screen_height), 10)  # Right orange line
        pygame.draw.line(self.screen, self.line_color, (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height), 10)  # Middle black line

        # Draw the boat based on the state
        if self.state == 0:
            # Boat on the left side
            self.screen.blit(self.boat_image, (self.screen_width // 4 - 100, self.screen_height // 2 - 50))  # Adjust for larger boat
        else:
            # Boat on the right side
            self.screen.blit(self.boat_image, (3 * self.screen_width // 4 - 100, self.screen_height // 2 - 50))  # Adjust for larger boat

        # Update the display
        pygame.display.flip()

    def close(self):
        pygame.quit()

  
# Grid World
class GridWorldEnv(gym.Env):
    def __init__(self, gamma=0.25):
        super(GridWorldEnv, self).__init__()
        self.grid_size = (6, 6)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size[0]),
            spaces.Discrete(self.grid_size[1])
        ))

        self.walls = [(i, 2) for i in [0, 1]] + [(i, 2) for i in [3, 4, 5]] + [(3, j) for j in [2, 3, 4]]
        self.terminal_states = {
            (5, 0): -50,  # Red (bad) state
            (1, 4): -50,  # Red (bad) state
            (5, 5): 100   # Green (good) state
        }
        self.default_reward = -1
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self.state = (0, 0)

        # Pygame setup
        pygame.init()
        self.window_size = 600  # 600x600 window size
        self.grid_pixel_size = self.window_size // self.grid_size[0]
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("GridWorld")

        # Load player (robot) image
        self.robot_image = pygame.image.load('package_RL_JBM/robot.png.jpg')
        # Scale it to fit the grid
        self.robot_image = pygame.transform.scale(self.robot_image, (self.grid_pixel_size, self.grid_pixel_size))

        # Colors
        self.wall_color = (0, 45, 98)  # Dark blue: #002D62
        self.bad_state_color = (136, 8, 8)  # Red: #880808
        self.good_state_color = (0, 106, 78)  # Green: #006A4E

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        if self.rng.random() < self.gamma:
            action = self.rng.choice([0, 1, 2, 3])
        
        i, j = self.state

        if action == 0:
            next_state = (max(i - 1, 0), j)
        elif action == 1:
            next_state = (i, min(j + 1, self.grid_size[1] - 1))
        elif action == 2:
            next_state = (min(i + 1, self.grid_size[0] - 1), j)
        elif action == 3:
            next_state = (i, max(j - 1, 0))
        
        if next_state in self.walls:
            next_state = self.state
        
        if next_state in self.terminal_states:
            reward = self.terminal_states[next_state]
            done = True
        else:
            reward = self.default_reward
            done = False

        self.state = next_state
        return self.state, reward, done, {}

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # White background

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.grid_pixel_size, i * self.grid_pixel_size, self.grid_pixel_size, self.grid_pixel_size)
                
                # Draw walls (dark blue)
                if (i, j) in self.walls:
                    pygame.draw.rect(self.screen, self.wall_color, rect)

                # Draw terminal states (red for bad states, green for good states)
                elif (i, j) in self.terminal_states:
                    color = self.good_state_color if self.terminal_states[(i, j)] > 0 else self.bad_state_color
                    pygame.draw.rect(self.screen, color, rect)

                # Draw the agent as a robot image
                elif (i, j) == self.state:
                    self.screen.blit(self.robot_image, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()

    def close(self):
        pygame.quit()