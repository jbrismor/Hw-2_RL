import gym
from gym import spaces
import numpy as np
import pygame

class BoatEnvSimulator(gym.Env):
    metadata = {'render_modes': [None]}

    def __init__(self, east_wind_prob=0.7, west_wind_prob=0.3, episodes=100, steps=1000, seed=None):
        super().__init__()
        assert east_wind_prob + west_wind_prob <= 1, 'Invalid wind probabilities'
        self.east_wind_prob = east_wind_prob
        self.west_wind_prob = west_wind_prob
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.episodes = episodes
        self.steps = steps
        self.transition_counts = {}
        self.state_rewards = {}
        self.rng = np.random.default_rng(seed)
        self.prob_wind = [east_wind_prob, west_wind_prob, 1 - east_wind_prob - west_wind_prob]
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
        return self.state, self.get_info(0)

    def step(self, action):
        wind = self.rng.choice([1, -1, 0], p=self.prob_wind)  # Randomly choose wind direction or no wind
        move_direction = 1 if action == 1 else -1  # 1 for right, -1 for left
        reward = 0

        # Determine next state and calculate reward based on the current state, action, and wind
        if (wind == move_direction or wind == 0):  # Move if wind and motor direction align or no wind
            if self.state == 0 and action == 1:
                self.state = 1  # Move right from left
                reward = 2
            elif self.state == 1 and action == 0:
                self.state = 0  # Move left from right
                reward = 2
        else:
            # Handling no movement due to opposing wind and motor direction
            if self.state == 0:
                reward = 1 if action == 0 else 0  # Hitting left wall or staying due to wind
            elif self.state == 1:
                reward = 4 if action == 1 else 3  # Hitting right wall or staying due to wind

        terminated = False
        truncated = False
        self.render()
        return self.state, reward, terminated, truncated, self.get_info(wind)
    
    def get_transitions(self, state, action):
        transitions = []
        action_direction = 1 if action == 1 else -1  # 1 for right, -1 for left

        # Iterate over possible wind effects
        for wind, prob in zip([1, -1, 0], self.prob_wind):
            move_direction = action_direction
            if wind == move_direction or wind == 0:  # Movement happens
                new_state = 1 if state == 0 and action == 1 else 0 if state == 1 and action == 0 else state
            else:
                new_state = state  # No movement due to opposing wind

            # Determine reward based on new state, current state, action, and wind
            if new_state != state:  # Successfully moved to the other state
                reward = 2
            else:
                if state == 0:
                    reward = 1 if action == 0 else 0  # No move or hit the wall
                elif state == 1:
                    reward = 4 if action == 1 else 3  # Hit the wall or no move

            transitions.append((prob, new_state, reward))
        
        return transitions

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

    def run_simulation(self):
        for _ in range(self.episodes):
            state, _ = self.reset()
            for _ in range(self.steps):
                action = np.random.choice([0, 1], p=[0.5, 0.5])
                next_state, reward, terminated, truncated, _ = self.step(action)
                transition_key = (state, action, next_state)
                reward_key = (state, action, next_state, reward)
                if transition_key not in self.transition_counts:
                    self.transition_counts[transition_key] = 0
                self.transition_counts[transition_key] += 1
                if reward_key not in self.state_rewards:
                    self.state_rewards[reward_key] = 0
                self.state_rewards[reward_key] += 1
                state = next_state
                if terminated or truncated:
                    break

    def close(self):
        pygame.quit()

  
# Grid World
# Grid World
class GridWorldEnv(gym.Env):
    metadata = {'render_modes': [None]}

    def __init__(self, gamma=0.25, episodes=10000, steps=1000, seed=None):
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
        self.episodes = episodes
        self.steps = steps
        self.rng = np.random.default_rng(seed)
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

    def get_transitions(self, state, action):
        transitions = []
        i, j = state
        directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]  # Up, Right, Down, Left
        next_state = directions[action]
        if next_state[0] < 0 or next_state[0] >= self.env.grid_size[0] or next_state[1] < 0 or next_state[1] >= self.env.grid_size[1]:
            next_state = state  # Handle out-of-bounds by staying in the current state
        if next_state in self.env.walls:
            next_state = state  # Handle wall collision by staying in the current state

        if next_state in self.env.terminal_states:
            reward = self.env.terminal_states[next_state]
        else:
            reward = self.env.default_reward

        # The probability of taking the intended action unless influenced by randomness (gamma)
        prob = 1 - self.env.gamma + (self.env.gamma / self.env.action_space.n) if action == action else (self.env.gamma / self.env.action_space.n)
        transitions.append((prob, next_state, reward))

        # Adding random action influence
        for a in range(self.env.action_space.n):
            if a != action:
                alternative_next_state = directions[a]
                if alternative_next_state[0] < 0 or alternative_next_state[0] >= self.env.grid_size[0] or alternative_next_state[1] < 0 or alternative_next_state[1] >= self.env.grid_size[1]:
                    alternative_next_state = state  # Handle out-of-bounds by staying in the current state
                if alternative_next_state in self.env.walls:
                    alternative_next_state = state  # Handle wall collision by staying in the current state

                transitions.append((self.env.gamma / self.env.action_space.n, alternative_next_state, self.env.default_reward))

        return transitions


    def get_next_state_reward(self, state, action):
        # This function computes the next state and reward for a given action without randomness
        i, j = state
        if action == 0:
            next_state = (max(i - 1, 0), j)
        elif action == 1:
            next_state = (i, min(j + 1, self.grid_size[1] - 1))
        elif action == 2:
            next_state = (min(i + 1, self.grid_size[0] - 1), j)
        elif action == 3:
            next_state = (i, max(j - 1, 0))

        if next_state in self.walls:
            next_state = state

        if next_state in self.terminal_states:
            reward = self.terminal_states[next_state]
        else:
            reward = self.default_reward

        return next_state, reward

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


# Geo Search