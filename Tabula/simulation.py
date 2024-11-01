import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Boat Environment Class
class BoatEnv(gym.Env):
    # Metadata specifying no specific rendering modes
    metadata = {'render_modes': ['None']}

    def __init__(self, east_wind_prob=0.7, west_wind_prob=0.3, episodes=100, steps=1000, seed=None, render=False):
        """
        Initializes the BoatEnv environment with parameters for wind probabilities, 
        number of episodes, steps per episode, and optional rendering.

        Args:
            east_wind_prob (float): Probability of east wind affecting movement.
            west_wind_prob (float): Probability of west wind affecting movement.
            episodes (int): Number of episodes in simulation.
            steps (int): Steps allowed per episode.
            seed (int, optional): Random seed for reproducibility.
            render (bool, optional): Whether to enable rendering.
        """
        super().__init__()
        
        # Wind probabilities and setup
        self.east_wind_prob = east_wind_prob
        self.west_wind_prob = west_wind_prob
        self.observation_space = spaces.Discrete(2)  # Two states
        self.action_space = spaces.Discrete(2)  # Two possible actions
        self.episodes = episodes
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.prob_wind = [east_wind_prob, west_wind_prob, 1 - east_wind_prob - west_wind_prob]  # Probabilities of wind directions
        self.rendering_enabled = render
        
        # Initialize Pygame if rendering is enabled
        pygame.init()
        self.screen_width, self.screen_height = 600, 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Boat Environment")
        self.bg_color = (173, 216, 230)  # Background color
        self.line_color = (0, 0, 0)  # Mid-line color
        self.end_line_color = (255, 165, 0)  # End line color
        self.boat_image = pygame.image.load('Tabula/boat.jpg')
        self.boat_image = pygame.transform.scale(self.boat_image, (200, 100))  # Resized boat image

    def get_info(self, wind):
        """
        Provides wind information based on wind direction.

        Args:
            wind (int): Wind direction, where 1 is East, -1 is West, and 0 is no wind.

        Returns:
            dict: Wind direction information.
        """
        direction = {1: 'East Wind', -1: 'West Wind', 0: 'No Wind'}
        return {'Wind': direction[wind]}

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            int: The initial state (0).
        """
        self.state = 0
        self.render()  # Render initial position if enabled
        return self.state  # Return the initial state

    def step(self, action):
        """
        Executes a step in the environment based on the chosen action.

        Args:
            action (int): The action to take, where 1 is move right and 0 is move left.

        Returns:
            tuple: Next state, reward, terminated flag, truncated flag.
        """
        wind = self.rng.choice([1, -1, 0], p=self.prob_wind)  # Randomly selects wind direction based on probabilities
        move_direction = 1 if action == 1 else -1
        reward = 0

        # Determine next state and reward
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
        return self.state, reward, terminated, truncated  # Returns without additional info

    def get_transitions(self, state, action):
        """
        Provides possible transitions for a given state and action with associated rewards and probabilities.

        Args:
            state (int): Current state.
            action (int): Chosen action.

        Returns:
            list: Possible transitions with (probability, next_state, reward).
        """
        transitions = []
        action_direction = 1 if action == 1 else -1

        # Define transitions for each wind scenario
        for wind, prob in zip([1, -1, 0], self.prob_wind):
            if wind == action_direction or wind == 0:
                new_state = 1 if state == 0 and action == 1 else 0 if state == 1 and action == 0 else state
            else:
                new_state = state  # No movement due to opposing wind

            reward = 2 if new_state != state else (1 if state == 0 and action == 0 else (4 if state == 1 and action == 1 else 3))
            transitions.append((prob, new_state, reward))
        
        return transitions

    def render(self, mode='human'):
        """
        Renders the environment visually using Pygame.

        Args:
            mode (str, optional): Rendering mode (default is 'human').
        """
        if not self.rendering_enabled:
            return
        
        # Clear screen and set background color
        self.screen.fill(self.bg_color)
        
        # Draw boundary and mid lines
        pygame.draw.line(self.screen, self.end_line_color, (0, 0), (0, self.screen_height), 10)
        pygame.draw.line(self.screen, self.end_line_color, (self.screen_width - 10, 0), (self.screen_width - 10, self.screen_height), 10)
        pygame.draw.line(self.screen, self.line_color, (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height), 10)

        # Display boat image based on state
        if self.state == 0:
            self.screen.blit(self.boat_image, (self.screen_width // 4 - 100, self.screen_height // 2 - 50))
        else:
            self.screen.blit(self.boat_image, (3 * self.screen_width // 4 - 100, self.screen_height // 2 - 50))
        
        pygame.display.flip()  # Update display

    def run_simulation(self):
        """
        Runs a full simulation across the specified number of episodes and steps.
        """
        for _ in range(self.episodes):
            state = self.reset()
            for _ in range(self.steps):
                action = np.random.choice([0, 1], p=[0.5, 0.5])  # Randomly choose an action
                next_state, reward, terminated, truncated = self.step(action)
                state = next_state
                if terminated or truncated:
                    break

    def close(self):
        """
        Closes the Pygame window and quits Pygame.
        """
        pygame.quit()


# GridWorld Environment Class
class GridWorldEnv(gym.Env):
    # Metadata specifying human rendering mode
    metadata = {'render_modes': ["human"]}

    def __init__(self, gamma=0.25, episodes=10000, steps=1000, seed=None, render=False):
        """
        Initializes the GridWorld environment with grid settings, wind probability, 
        number of episodes, steps, and rendering options.

        Args:
            gamma (float): Probability of random action for environment noise.
            episodes (int): Number of episodes for the simulation.
            steps (int): Maximum steps per episode.
            seed (int, optional): Random seed for reproducibility.
            render (bool, optional): Enable visual rendering with Pygame.
        """
        super().__init__()
        
        # Environment configuration
        self.grid_size = (6, 6)
        self.action_space = spaces.Discrete(4)  # Four possible actions (Up, Right, Down, Left)
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        self.gamma = gamma
        self.episodes = episodes
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.state = (0, 0)  # Starting state
        self.rendering_enabled = render

        # Define walls and terminal states
        self.walls = [(i, 2) for i in [0, 1]] + [(i, 2) for i in [3, 4, 5]] + [(3, j) for j in [2, 3, 4]]
        self.terminal_states = {
            (5, 0): -50,  # Negative terminal state
            (1, 4): -50,  # Another negative terminal state
            (5, 5): 100   # Positive terminal state
        }
        self.default_reward = -1  # Default reward for non-terminal moves

        # Initialize Pygame for rendering, if enabled
        pygame.init()
        self.window_size = 600
        self.grid_pixel_size = self.window_size // self.grid_size[0]
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("GridWorld")
        self.robot_image = pygame.image.load('Tabula/robot.png.jpg')
        self.robot_image = pygame.transform.scale(self.robot_image, (self.grid_pixel_size, self.grid_pixel_size))

        # Colors for walls, bad and good terminal states
        self.wall_color = (0, 45, 98)
        self.bad_state_color = (136, 8, 8)
        self.good_state_color = (0, 106, 78)

    def state_to_index(self, state):
        """
        Converts (i, j) state format to a unique integer index.

        Args:
            state (tuple): (i, j) grid coordinates.

        Returns:
            int: Flattened index representing the state.
        """
        i, j = state
        return i * self.grid_size[1] + j

    def index_to_state(self, index):
        """
        Converts integer index back to (i, j) grid coordinates.

        Args:
            index (int): Flattened state index.

        Returns:
            tuple: (i, j) grid coordinates.
        """
        return (index // self.grid_size[1], index % self.grid_size[1])

    def reset(self):
        """
        Resets the environment to the starting state.

        Returns:
            int: Flattened index of the initial state.
        """
        self.state = (0, 0)
        return self.state_to_index(self.state)

    def step(self, action):
        """
        Takes an action in the environment, applying the effects of noise 
        with probability gamma, and returns the result.

        Args:
            action (int): Action to take (0=Up, 1=Right, 2=Down, 3=Left).

        Returns:
            tuple: (next_state_index, reward, done, info) where next_state_index is the 
                   integer index of the new state, reward is the obtained reward, 
                   done indicates if the episode has ended, and info is an empty dictionary.
        """
        i, j = self.state
        directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]  # Action mapping

        # Environment noise: random action selection with probability gamma
        if self.rng.random() < self.gamma:
            action = self.rng.integers(0, 4)

        next_state = directions[action]

        # Check if next state is valid
        if next_state in self.walls or not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]):
            next_state = self.state  # Stay in current state if invalid move

        reward = self.terminal_states.get(next_state, self.default_reward)
        done = next_state in self.terminal_states
        self.state = next_state  # Update current state
        return self.state_to_index(self.state), reward, done, {}

    def get_transitions(self, state, action):
        """
        Returns possible transitions for a state-action pair, listing all reachable states 
        with associated rewards and probabilities.

        Args:
            state (int): Current state as integer index.
            action (int): Chosen action.

        Returns:
            list: List of transitions (probability, next_state, reward).
        """
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
        """
        Renders the GridWorld environment using Pygame.

        Args:
            mode (str, optional): Rendering mode (default is 'human').
        """
        if not self.rendering_enabled:
            return

        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid elements
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

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Draw grid outline

        pygame.display.flip()  # Update display

    def close(self):
        """
        Closes the Pygame window and exits Pygame.
        """
        pygame.quit()

# Geosearch Environment Class
class GeosearchEnv(gym.Env):
    # Metadata specifying human rendering mode
    metadata = {'render_modes': ['human']}

    def __init__(self, A=0.75, seed=None, render=False):
        """
        Initializes the Geosearch environment with parameters for Gaussian distributions,
        grid size, and rendering options.

        Args:
            A (float): Weight factor for combining two Gaussian distributions.
            seed (int, optional): Random seed for reproducibility.
            render (bool, optional): Enable visual rendering with Pygame.
        """
        super(GeosearchEnv, self).__init__()

        # Environment configuration
        self.grid_size = 25
        self.action_space = spaces.Discrete(4)  # Four possible actions: Up, Down, Left, Right
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # Parameters for Gaussian reward distributions
        self.mu1 = np.array([20, 20])
        self.sigma1 = np.array([1, 1])
        self.rho1 = 0.25
        self.mu2 = np.array([10, 10])
        self.sigma2 = np.array([1, 1])
        self.rho2 = -0.25
        self.A = A

        # Compute reward matrix based on Gaussian distributions
        self.reward_matrix = self.compute_rewards()
        self.state = (0, 0)  # Initial position

        # Initialize Pygame if rendering is enabled
        pygame.init()
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Geosearch Environment")
        self.rendering_enabled = render

    def gaussian_distribution(self, x, y, mu, sigma, rho):
        """
        Calculates the probability density for a 2D Gaussian distribution at (x, y).

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            mu (array): Mean vector of the Gaussian.
            sigma (array): Standard deviations for x and y.
            rho (float): Correlation coefficient.

        Returns:
            float: Probability density value at the point (x, y).
        """
        x_mu = x - mu[0]
        y_mu = y - mu[1]
        sigma_x, sigma_y = sigma
        rho_term = 2 * rho * x_mu * y_mu / (sigma_x * sigma_y)
        exponent = -1 / (2 * (1 - rho ** 2)) * ((x_mu ** 2 / sigma_x ** 2) - rho_term + (y_mu ** 2 / sigma_y ** 2))
        return (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))) * np.exp(exponent)

    def compute_rewards(self):
        """
        Generates a reward matrix by combining two Gaussian distributions.

        Returns:
            np.array: 2D array of rewards for each position in the grid.
        """
        rewards = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                f1 = self.gaussian_distribution(x, y, self.mu1, self.sigma1, self.rho1)
                f2 = self.gaussian_distribution(x, y, self.mu2, self.sigma2, self.rho2)
                rewards[x, y] = self.A * f1 + (1 - self.A) * f2  # Weighted sum of distributions
        return rewards

    def state_to_index(self, state):
        """
        Converts (x, y) grid position to a unique integer index.

        Args:
            state (tuple): (x, y) grid coordinates.

        Returns:
            int: Flattened index representing the state.
        """
        x, y = state
        return x * self.grid_size + y

    def index_to_state(self, index):
        """
        Converts integer index back to (x, y) grid coordinates.

        Args:
            index (int): Flattened state index.

        Returns:
            tuple: (x, y) grid coordinates.
        """
        return (index // self.grid_size, index % self.grid_size)

    def reset(self):
        """
        Resets the environment to a random starting position.

        Returns:
            int: Flattened index of the initial random state.
        """
        self.state = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        return self.state_to_index(self.state)

    def step(self, action):
        """
        Takes an action in the environment, moving the agent and returning the result.

        Args:
            action (int): Action to take (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            tuple: (next_state_index, reward, done, info) where next_state_index is the 
                   integer index of the new state, reward is the obtained reward, 
                   done indicates if the episode has ended, and info is an empty dictionary.
        """
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
        """
        Provides possible transitions for a given state-action pair, listing all reachable 
        states with associated rewards and probabilities.

        Args:
            state (int): Current state as integer index.
            action (int): Chosen action.

        Returns:
            list: List of transitions (probability, next_state, reward).
        """
        x, y = self.index_to_state(state)
        directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]  # Up, Down, Left, Right
        transitions = []
        
        for idx, (next_x, next_y) in enumerate(directions):
            if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                next_state = self.state_to_index((next_x, next_y))
            else:
                next_state = state  # Stay in current state if out of bounds

            reward = self.reward_matrix[next_x, next_y] if next_state != state else self.reward_matrix[x, y]
            prob = 1.0 if idx == action else 0.0
            transitions.append((prob, next_state, reward))

        return transitions

    def render(self, mode='human'):
        """
        Renders the Geosearch environment with color intensity based on reward values 
        and highlights the agent’s current position.

        Args:
            mode (str, optional): Rendering mode (default is 'human').
        """
        if not self.rendering_enabled:
            return
        
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw reward-based color intensity for each grid cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color_intensity = int(255 * self.reward_matrix[x, y] / np.max(self.reward_matrix))
                pygame.draw.rect(self.screen, (0, 0, color_intensity), 
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        
        # Draw agent position in red
        agent_pos = (self.state[0] * self.cell_size, self.state[1] * self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), (agent_pos[0], agent_pos[1], self.cell_size, self.cell_size))
        
        pygame.display.flip()  # Update display

    def close(self):
        """
        Closes the Pygame window and exits Pygame.
        """
        pygame.quit()
