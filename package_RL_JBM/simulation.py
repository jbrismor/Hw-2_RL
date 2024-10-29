import gymnasium as gym
from gym import spaces
import numpy as np
import pygame

class BoatEnvSimulator(gym.Env):
    metadata = {'render_modes': [None]}

    def __init__(self, east_wind_prob=0.7, west_wind_prob=0.3, episodes=100, steps=1000, seed=None):
        super().__init__()
        self.east_wind_prob = east_wind_prob
        self.west_wind_prob = west_wind_prob
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.episodes = episodes
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.prob_wind = [east_wind_prob, west_wind_prob, 1 - east_wind_prob - west_wind_prob]
        
        # Pygame setup
        pygame.init()
        self.screen_width, self.screen_height = 600, 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Boat Environment")
        self.bg_color = (173, 216, 230)
        self.line_color = (0, 0, 0)
        self.end_line_color = (255, 165, 0)
        self.boat_image = pygame.image.load('package_RL_JBM/boat.jpg')
        self.boat_image = pygame.transform.scale(self.boat_image, (200, 100))

    def get_info(self, wind):
        direction = {1: 'East Wind', -1: 'West Wind', 0: 'No Wind'}
        return {'Wind': direction[wind]}

    def reset(self):
        self.state = 0
        self.render()
        return self.state  # Return only the state, without the additional info

    def step(self, action):
        wind = self.rng.choice([1, -1, 0], p=self.prob_wind)
        move_direction = 1 if action == 1 else -1
        reward = 0

        # Calculate next state and reward based on action and wind
        if wind == move_direction or wind == 0:
            if self.state == 0 and action == 1:
                self.state = 1
                reward = 2
            elif self.state == 1 and action == 0:
                self.state = 0
                reward = 2
        else:
            reward = 1 if (self.state == 0 and action == 0) else (4 if (self.state == 1 and action == 1) else 0)

        terminated = False
        truncated = False
        self.render()
        return self.state, reward, terminated, truncated  # No info returned here

    def get_transitions(self, state, action):
        transitions = []
        action_direction = 1 if action == 1 else -1

        # Define transitions based on wind probabilities
        for wind, prob in zip([1, -1, 0], self.prob_wind):
            if wind == action_direction or wind == 0:
                new_state = 1 if state == 0 and action == 1 else 0 if state == 1 and action == 0 else state
            else:
                new_state = state  # No movement due to opposing wind

            reward = 2 if new_state != state else (1 if state == 0 and action == 0 else (4 if state == 1 and action == 1 else 3))
            transitions.append((prob, new_state, reward))
        
        return transitions

    def render(self, mode='human'):
        self.screen.fill(self.bg_color)
        
        # Draw boundary lines
        pygame.draw.line(self.screen, self.end_line_color, (0, 0), (0, self.screen_height), 10)
        pygame.draw.line(self.screen, self.end_line_color, (self.screen_width - 10, 0), (self.screen_width - 10, self.screen_height), 10)
        pygame.draw.line(self.screen, self.line_color, (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height), 10)

        # Position the boat based on its current state
        if self.state == 0:
            self.screen.blit(self.boat_image, (self.screen_width // 4 - 100, self.screen_height // 2 - 50))
        else:
            self.screen.blit(self.boat_image, (3 * self.screen_width // 4 - 100, self.screen_height // 2 - 50))
        
        pygame.display.flip()

    def run_simulation(self):
        for _ in range(self.episodes):
            state = self.reset()
            for _ in range(self.steps):
                action = np.random.choice([0, 1], p=[0.5, 0.5])
                next_state, reward, terminated, truncated = self.step(action)
                state = next_state
                if terminated or truncated:
                    break

    def close(self):
        pygame.quit()

# class BoatEnvSimulator(gym.Env):
#     metadata = {'render_modes': [None]}

#     def __init__(self, east_wind_prob=0.7, west_wind_prob=0.3, episodes=100, steps=1000, seed=None):
#         super().__init__()
#         self.east_wind_prob = east_wind_prob
#         self.west_wind_prob = west_wind_prob
#         self.observation_space = spaces.Discrete(2)
#         self.action_space = spaces.Discrete(2)
#         self.episodes = episodes
#         self.steps = steps
#         self.rng = np.random.default_rng(seed)
#         self.prob_wind = [east_wind_prob, west_wind_prob, 1 - east_wind_prob - west_wind_prob]
        
#         # Pygame setup
#         pygame.init()
#         self.screen_width, self.screen_height = 600, 400
#         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#         pygame.display.set_caption("Boat Environment")
#         self.bg_color = (173, 216, 230)
#         self.line_color = (0, 0, 0)
#         self.end_line_color = (255, 165, 0)
#         self.boat_image = pygame.image.load('package_RL_JBM/boat.jpg')
#         self.boat_image = pygame.transform.scale(self.boat_image, (200, 100))

#     def get_info(self, wind):
#         direction = {1: 'East Wind', -1: 'West Wind', 0: 'No Wind'}
#         return {'Wind': direction[wind]}

#     def reset(self):
#         self.state = 0
#         self.render()
#         return self.state, self.get_info(0)

#     def step(self, action):
#         wind = self.rng.choice([1, -1, 0], p=self.prob_wind)
#         move_direction = 1 if action == 1 else -1
#         reward = 0

#         # Calculate next state and reward based on action and wind
#         if wind == move_direction or wind == 0:
#             if self.state == 0 and action == 1:
#                 self.state = 1
#                 reward = 2
#             elif self.state == 1 and action == 0:
#                 self.state = 0
#                 reward = 2
#         else:
#             reward = 1 if (self.state == 0 and action == 0) else (4 if (self.state == 1 and action == 1) else 0)

#         terminated = False
#         truncated = False
#         self.render()
#         return self.state, reward, terminated, truncated, self.get_info(wind)

#     def get_transitions(self, state, action):
#         transitions = []
#         action_direction = 1 if action == 1 else -1

#         # Define transitions based on wind probabilities
#         for wind, prob in zip([1, -1, 0], self.prob_wind):
#             if wind == action_direction or wind == 0:
#                 new_state = 1 if state == 0 and action == 1 else 0 if state == 1 and action == 0 else state
#             else:
#                 new_state = state  # No movement due to opposing wind

#             reward = 2 if new_state != state else (1 if state == 0 and action == 0 else (4 if state == 1 and action == 1 else 3))
#             transitions.append((prob, new_state, reward))
        
#         return transitions

#     def render(self, mode='human'):
#         self.screen.fill(self.bg_color)
        
#         # Draw boundary lines
#         pygame.draw.line(self.screen, self.end_line_color, (0, 0), (0, self.screen_height), 10)
#         pygame.draw.line(self.screen, self.end_line_color, (self.screen_width - 10, 0), (self.screen_width - 10, self.screen_height), 10)
#         pygame.draw.line(self.screen, self.line_color, (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height), 10)

#         # Position the boat based on its current state
#         if self.state == 0:
#             self.screen.blit(self.boat_image, (self.screen_width // 4 - 100, self.screen_height // 2 - 50))
#         else:
#             self.screen.blit(self.boat_image, (3 * self.screen_width // 4 - 100, self.screen_height // 2 - 50))
        
#         pygame.display.flip()

#     def run_simulation(self):
#         for _ in range(self.episodes):
#             state, _ = self.reset()
#             for _ in range(self.steps):
#                 action = np.random.choice([0, 1], p=[0.5, 0.5])
#                 next_state, reward, terminated, truncated, _ = self.step(action)
#                 state = next_state
#                 if terminated or truncated:
#                     break

#     def close(self):
#         pygame.quit()


# class GridWorldEnv(gym.Env):
    # metadata = {'render_modes': [None]}

    # def __init__(self, gamma=0.25, episodes=10000, steps=1000, seed=None):
    #     super().__init__()
    #     self.grid_size = (6, 6)
    #     self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
    #     self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
    #     self.gamma = gamma
    #     self.episodes = episodes
    #     self.steps = steps
    #     self.rng = np.random.default_rng(seed)
    #     self.state = (0, 0)

    #     # Define walls and terminal states
    #     self.walls = [(i, 2) for i in [0, 1]] + [(i, 2) for i in [3, 4, 5]] + [(3, j) for j in [2, 3, 4]]
    #     self.terminal_states = {
    #         (5, 0): -50,  # Negative terminal state
    #         (1, 4): -50,  # Another negative terminal state
    #         (5, 5): 100   # Positive terminal state
    #     }
    #     self.default_reward = -1

    #     # Pygame setup for rendering
    #     pygame.init()
    #     self.window_size = 600  # 600x600 window size
    #     self.grid_pixel_size = self.window_size // self.grid_size[0]
    #     self.screen = pygame.display.set_mode((self.window_size, self.window_size))
    #     pygame.display.set_caption("GridWorld")
    #     self.robot_image = pygame.image.load('package_RL_JBM/robot.png.jpg')
    #     self.robot_image = pygame.transform.scale(self.robot_image, (self.grid_pixel_size, self.grid_pixel_size))

    #     # Define colors
    #     self.wall_color = (0, 45, 98)       # Dark blue for walls
    #     self.bad_state_color = (136, 8, 8)  # Red for negative terminal states
    #     self.good_state_color = (0, 106, 78)  # Green for positive terminal states

    # def state_to_index(self, state):
    #     i, j = state
    #     return i * self.grid_size[1] + j

    # def index_to_state(self, index):
    #     return (index // self.grid_size[1], index % self.grid_size[1])

    # def reset(self):
    #     self.state = (0, 0)
    #     return self.state_to_index(self.state)

    # def step(self, action):
    #     i, j = self.state
    #     # Map actions to directions
    #     directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
    #     next_state = directions[action]
        
    #     # Check for walls and boundaries
    #     if next_state in self.walls or not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]):
    #         next_state = self.state  # Stay in the current state

    #     # Get reward and check if the state is terminal
    #     reward = self.terminal_states.get(next_state, self.default_reward)
    #     done = next_state in self.terminal_states
    #     self.state = next_state
    #     return self.state_to_index(self.state), reward, done, {}

    # def get_transitions(self, state, action):
    #     transitions = []
    #     i, j = self.index_to_state(state)
    #     directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
        
    #     for dir_index, (next_i, next_j) in enumerate(directions):
    #         # Verify if the next state is a wall or out of bounds
    #         if (next_i, next_j) in self.walls or not (0 <= next_i < self.grid_size[0] and 0 <= next_j < self.grid_size[1]):
    #             next_state = state
    #         else:
    #             next_state = self.state_to_index((next_i, next_j))
            
    #         # Determine reward and probability
    #         reward = self.terminal_states.get(self.index_to_state(next_state), self.default_reward)
    #         prob = 1.0 if dir_index == action else 0.0
    #         transitions.append((prob, next_state, reward))
        
    #     return transitions

    # def render(self, mode='human'):
    #     self.screen.fill((255, 255, 255))  # White background
        
    #     for i in range(self.grid_size[0]):
    #         for j in range(self.grid_size[1]):
    #             rect = pygame.Rect(j * self.grid_pixel_size, i * self.grid_pixel_size, self.grid_pixel_size, self.grid_pixel_size)
                
    #             # Draw walls and terminal states
    #             if (i, j) in self.walls:
    #                 pygame.draw.rect(self.screen, self.wall_color, rect)
    #             elif (i, j) in self.terminal_states:
    #                 color = self.good_state_color if self.terminal_states[(i, j)] > 0 else self.bad_state_color
    #                 pygame.draw.rect(self.screen, color, rect)
    #             elif (i, j) == self.state:
    #                 self.screen.blit(self.robot_image, rect)

    #             pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Grid lines
        
    #     pygame.display.flip()

    # def close(self):
    #     pygame.quit()

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': [None]}

    def __init__(self, gamma=0.25, episodes=10000, steps=1000, seed=None):
        super().__init__()
        self.grid_size = (6, 6)
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        self.gamma = gamma
        self.episodes = episodes
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.state = (0, 0)  # Initialize state as (i, j) tuple

        # Define walls and terminal states
        self.walls = [(i, 2) for i in [0, 1]] + [(i, 2) for i in [3, 4, 5]] + [(3, j) for j in [2, 3, 4]]
        self.terminal_states = {
            (5, 0): -50,  # Negative terminal state
            (1, 4): -50,  # Another negative terminal state
            (5, 5): 100   # Positive terminal state
        }
        self.default_reward = -1

        # Pygame setup for rendering
        pygame.init()
        self.window_size = 600
        self.grid_pixel_size = self.window_size // self.grid_size[0]
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("GridWorld")
        self.robot_image = pygame.image.load('package_RL_JBM/robot.png.jpg')
        self.robot_image = pygame.transform.scale(self.robot_image, (self.grid_pixel_size, self.grid_pixel_size))

        self.wall_color = (0, 45, 98)
        self.bad_state_color = (136, 8, 8)
        self.good_state_color = (0, 106, 78)

    def state_to_index(self, state):
        i, j = state
        return i * self.grid_size[1] + j

    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])

    def reset(self):
        self.state = (0, 0)  # Reset to (i, j) tuple format
        return self.state_to_index(self.state)  # Return integer index

    def step(self, action):
        i, j = self.state
        directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
        next_state = directions[action]

        if next_state in self.walls or not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]):
            next_state = self.state

        reward = self.terminal_states.get(next_state, self.default_reward)
        done = next_state in self.terminal_states
        self.state = next_state  # Update state as (i, j) tuple
        return self.state_to_index(self.state), reward, done, {}

    def get_transitions(self, state, action):
        transitions = []
        i, j = self.index_to_state(state)
        directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]

        for dir_index, (next_i, next_j) in enumerate(directions):
            if (next_i, next_j) in self.walls or not (0 <= next_i < self.grid_size[0] and 0 <= next_j < self.grid_size[1]):
                next_state = state
            else:
                next_state = self.state_to_index((next_i, next_j))

            reward = self.terminal_states.get((next_i, next_j), self.default_reward)
            prob = 1.0 if dir_index == action else 0.0
            transitions.append((prob, next_state, reward))
        
        return transitions

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.grid_pixel_size, i * self.grid_pixel_size, self.grid_pixel_size, self.grid_pixel_size)
                
                if (i, j) in self.walls:
                    pygame.draw.rect(self.screen, self.wall_color, rect)
                elif (i, j) in self.terminal_states:
                    color = self.good_state_color if self.terminal_states[(i, j)] > 0 else self.bad_state_color
                    pygame.draw.rect(self.screen, color, rect)
                elif (i, j) == self.state:
                    self.screen.blit(self.robot_image, rect)

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()

    def close(self):
        pygame.quit()

class GeosearchEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, A=0.75):
        super(GeosearchEnv, self).__init__()
        self.grid_size = 25
        self.action_space = spaces.Discrete(4)  # Actions: 0-Up, 1-Down, 2-Left, 3-Right
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # Gaussian distribution parameters
        self.mu1 = np.array([20, 20])
        self.sigma1 = np.array([1, 1])
        self.rho1 = 0.25
        self.mu2 = np.array([10, 10])
        self.sigma2 = np.array([1, 1])
        self.rho2 = -0.25
        self.A = A

        self.reward_matrix = self.compute_rewards()
        self.state = (0, 0)  # Starting position

        # Pygame setup
        pygame.init()
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Geosearch Environment")

    def gaussian_distribution(self, x, y, mu, sigma, rho):
        # Probability density function for a 2D Gaussian
        x_mu = x - mu[0]
        y_mu = y - mu[1]
        sigma_x, sigma_y = sigma
        rho_term = 2 * rho * x_mu * y_mu / (sigma_x * sigma_y)
        exponent = -1 / (2 * (1 - rho ** 2)) * ((x_mu ** 2 / sigma_x ** 2) - rho_term + (y_mu ** 2 / sigma_y ** 2))
        return (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))) * np.exp(exponent)

    def compute_rewards(self):
        # Compute a reward matrix based on Gaussian distributions
        rewards = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                f1 = self.gaussian_distribution(x, y, self.mu1, self.sigma1, self.rho1)
                f2 = self.gaussian_distribution(x, y, self.mu2, self.sigma2, self.rho2)
                rewards[x, y] = self.A * f1 + (1 - self.A) * f2
        return rewards

    def state_to_index(self, state):
        x, y = state
        return x * self.grid_size + y

    def index_to_state(self, index):
        return (index // self.grid_size, index % self.grid_size)

    def reset(self):
        # Randomize the starting position within the grid
        self.state = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        return self.state_to_index(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0 and y > 0:            # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:  # Down
            y += 1
        elif action == 2 and x > 0:          # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # Right
            x += 1
        self.state = (x, y)
        reward = self.reward_matrix[x, y]
        done = (x, y) == (self.grid_size - 1, self.grid_size - 1)  # Terminal condition
        return self.state_to_index(self.state), reward, done, {}

    def get_transitions(self, state, action):
        x, y = self.index_to_state(state)
        directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]  # Up, Down, Left, Right
        transitions = []
        
        for idx, (next_x, next_y) in enumerate(directions):
            # Check boundaries
            if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                next_state = self.state_to_index((next_x, next_y))
            else:
                next_state = state  # If out of bounds, stay in current state

            reward = self.reward_matrix[next_x, next_y] if next_state != state else self.reward_matrix[x, y]
            prob = 1.0 if idx == action else 0.0
            transitions.append((prob, next_state, reward))

        return transitions

    def render(self, mode='human'):
        # Rendering the environment with reward gradients and the agent's position
        self.screen.fill((0, 0, 0))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color_intensity = int(255 * self.reward_matrix[x, y] / np.max(self.reward_matrix))
                pygame.draw.rect(self.screen, (0, 0, color_intensity), 
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        # Draw agent
        agent_pos = (self.state[0] * self.cell_size, self.state[1] * self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), (agent_pos[0], agent_pos[1], self.cell_size, self.cell_size))
        pygame.display.flip()

    def close(self):
        pygame.quit()
