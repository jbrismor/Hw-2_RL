import numpy as np
import gym
from collections import defaultdict

class DynamicProgrammingSolver:
    def __init__(self, env, gamma=0.99, theta=0.01, use_value_iteration=False):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.use_value_iteration = use_value_iteration

        # Handle different observation spaces
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_space_size = env.observation_space.n
            self.discrete_observation = True
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.state_space_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            self.discrete_observation = False
        else:
            raise ValueError("Unsupported observation space type. Only Discrete and Box are supported.")
        
        # Initialize value and policy arrays
        self.V = np.zeros(self.state_space_size)
        self.policy = np.zeros(self.state_space_size, dtype=int)

    def get_transitions(self, state, action):
        """Retrieve transitions from the environment"""
        return self.env.get_transitions(state, action)

    def expected_return(self, state, action):
        """Compute expected return of taking action in state based on transition probabilities and rewards."""
        return sum(prob * (reward + self.gamma * self.V[self.state_to_index(next_state)])
                   for prob, next_state, reward in self.get_transitions(state, action))

    def state_to_index(self, state):
        """Convert a state tuple to a single index for environments with continuous spaces."""
        if self.discrete_observation:
            return state
        else:
            return state[0] * self.env.observation_space.shape[1] + state[1]

    def value_iteration(self):
        """Run Value Iteration algorithm to compute optimal policy."""
        while True:
            delta = 0
            for s in range(self.state_space_size):
                v = self.V[s]
                action_values = [self.expected_return(self.index_to_state(s), a) for a in range(self.env.action_space.n)]
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break
        for s in range(self.state_space_size):
            self.policy[s] = np.argmax([self.expected_return(self.index_to_state(s), a) for a in range(self.env.action_space.n)])

    def index_to_state(self, index):
        """Convert an index back to a state tuple."""
        if self.discrete_observation:
            return index
        else:
            row = index // self.env.observation_space.shape[1]
            col = index % self.env.observation_space.shape[1]
            return (row, col)

    def solve(self):
        """Solve the MDP using either Value Iteration or Policy Iteration."""
        if self.use_value_iteration:
            self.value_iteration()
        else:
            self.policy_iteration()
        return self.policy, self.V

    def policy_iteration(self):
        """Run Policy Iteration algorithm to compute optimal policy."""
        stable = False
        while not stable:
            self.policy_evaluation()
            stable = self.policy_improvement()

    def policy_evaluation(self):
        """Evaluate policy until convergence."""
        while True:
            delta = 0
            for s in range(self.state_space_size):
                v = self.V[s]
                self.V[s] = self.expected_return(self.index_to_state(s), self.policy[s])
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        """Improve policy by making it greedy with respect to the current value function."""
        policy_stable = True
        for s in range(self.state_space_size):
            old_action = self.policy[s]
            best_action = np.argmax([self.expected_return(self.index_to_state(s), a) for a in range(self.env.action_space.n)])
            self.policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

class MonteCarloSolver:
    def __init__(self, env, gamma=0.99, epsilon=0.1, exploring_starts=True, episodes=1000, max_steps_per_episode=100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.exploring_starts = exploring_starts
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.Returns = defaultdict(list)
        self.policy = {s: np.random.choice(self.env.action_space.n) for s in range(self.env.observation_space.n)}

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < self.max_steps_per_episode:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        return episode

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def run(self):
        for episode_num in range(self.episodes):
            if self.exploring_starts:
                state = self.env.observation_space.sample()
                action = self.env.action_space.sample()
                self.env.state = self.env.index_to_state(state)
                episode = self.generate_episode_with_start(state, action)
            else:
                episode = self.generate_episode()

            G = 0
            visited_state_action_pairs = set()

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action) not in visited_state_action_pairs:
                    self.Returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.Returns[(state, action)])
                    self.policy[state] = np.argmax(self.Q[state])
                    visited_state_action_pairs.add((state, action))

            if episode_num % 100 == 0:
                print(f"Policy after {episode_num} episodes: {self.policy}")

        return self.policy

    def generate_episode_with_start(self, start_state, start_action):
        episode = []
        state = start_state
        action = start_action
        done = False
        steps = 0

        while not done and steps < self.max_steps_per_episode:
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            action = self.select_action(state)
            steps += 1

        return episode
