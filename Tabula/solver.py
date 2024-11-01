import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


# DPSolver class
class DPSolver:
    def __init__(self, env, gamma=0.99, theta=0.01, use_value_iteration=False):
        """
        Initializes the DPSolver with parameters for discounting, convergence threshold,
        and choice between Value Iteration and Policy Iteration.

        Args:
            env (gym.Env): The environment to solve, assumed to implement discrete actions and states.
            gamma (float): Discount factor for future rewards.
            theta (float): Threshold for convergence.
            use_value_iteration (bool): Whether to use Value Iteration (True) or Policy Iteration (False).
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.use_value_iteration = use_value_iteration

        # Determine state space size based on environment's observation space
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_space_size = env.observation_space.n
            self.discrete_observation = True
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.state_space_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            self.discrete_observation = False
        else:
            raise ValueError("Unsupported observation space type. Only Discrete and Box are supported.")

        # Initialize value, action-value, and policy arrays
        self.V = np.zeros(self.state_space_size)
        self.Q = np.zeros((self.state_space_size, self.env.action_space.n))
        self.policy = np.random.choice(self.env.action_space.n, size=self.state_space_size)

        # Track history for convergence analysis
        self.V_history = []
        self.Q_history = []
        self.policy_history = []

    def get_transitions(self, state, action):
        """
        Retrieves transitions from the environment.

        Args:
            state (int or tuple): Current state.
            action (int): Action taken.

        Returns:
            list: List of (probability, next_state, reward) tuples.
        """
        return self.env.get_transitions(state, action)

    def expected_return(self, state, action):
        """
        Calculates the expected return of taking an action in a state based on
        transition probabilities and rewards.

        Args:
            state (int or tuple): Current state.
            action (int): Action taken.

        Returns:
            float: Expected return for taking the action in the state.
        """
        expected_return = 0
        for prob, next_state, reward in self.get_transitions(state, action):
            next_state_index = self.state_to_index(next_state)
            expected_return += prob * (reward + self.gamma * self.V[next_state_index])
        return expected_return

    def compute_Q(self):
        """
        Computes Q(s, a) for all state-action pairs based on the current value function V.
        """
        for s in range(self.state_space_size):
            state = self.index_to_state(s)
            for a in range(self.env.action_space.n):
                self.Q[s, a] = self.expected_return(state, a)

    def state_to_index(self, state):
        """
        Converts a state tuple to a single index for environments with continuous spaces.

        Args:
            state (int or tuple): State to convert.

        Returns:
            int: Flattened index representing the state.
        """
        if self.discrete_observation:
            return state
        else:
            return state[0] * self.env.observation_space.shape[1] + state[1]

    def index_to_state(self, index):
        """
        Converts an index back to a state tuple.

        Args:
            index (int): Flattened state index.

        Returns:
            int or tuple: Original state format.
        """
        if self.discrete_observation:
            return index
        else:
            row = index // self.env.observation_space.shape[1]
            col = index % self.env.observation_space.shape[1]
            return (row, col)

    def value_iteration(self):
        """
        Runs the Value Iteration algorithm to compute the optimal policy.
        """
        iteration = 0
        while True:
            delta = 0
            self.V_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())
            for s in range(self.state_space_size):
                v = self.V[s]
                action_values = []
                state = self.index_to_state(s)
                for a in range(self.env.action_space.n):
                    action_value = self.expected_return(state, a)
                    action_values.append(action_value)
                    self.Q[s, a] = action_value  # Update Q(s,a)
                self.V[s] = max(action_values)
                self.policy[s] = np.argmax(action_values)
                delta = max(delta, abs(v - self.V[s]))
            self.Q_history.append(self.Q.copy())
            iteration += 1
            if delta < self.theta:
                break

    def solve(self):
        """
        Solves the MDP using either Value Iteration or Policy Iteration.

        Returns:
            tuple: Optimal policy, value function, and action-value function.
        """
        if self.use_value_iteration:
            self.value_iteration()
        else:
            self.policy_iteration()
        self.compute_Q()  # Final computation of Q-values
        return self.policy, self.V, self.Q

    def policy_iteration(self):
        """
        Runs the Policy Iteration algorithm to compute the optimal policy.
        """
        stable = False
        while not stable:
            self.V_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())
            self.policy_evaluation()
            stable = self.policy_improvement()
            self.compute_Q()
            self.Q_history.append(self.Q.copy())

    def policy_evaluation(self):
        """
        Evaluates the policy until convergence.
        """
        while True:
            delta = 0
            for s in range(self.state_space_size):
                v = self.V[s]
                state = self.index_to_state(s)
                a = self.policy[s]
                self.V[s] = self.expected_return(state, a)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        """
        Improves the policy by making it greedy with respect to the current value function.

        Returns:
            bool: Whether the policy is stable (i.e., no changes made to the policy).
        """
        policy_stable = True
        for s in range(self.state_space_size):
            old_action = self.policy[s]
            action_values = []
            state = self.index_to_state(s)
            for a in range(self.env.action_space.n):
                action_value = self.expected_return(state, a)
                action_values.append(action_value)
                self.Q[s, a] = action_value  # Update Q(s,a)
            best_action = np.argmax(action_values)
            self.policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable


# MCSolver class
class MCSolver:
    def __init__(self, env, gamma=0.99, epsilon=0.1, exploring_starts=True, episodes=1000, max_steps_per_episode=100):
        """
        Initializes the MCSolver with parameters for discounting, exploration, and episode settings.

        Args:
            env (gym.Env): The environment to solve.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Probability for epsilon-greedy action selection.
            exploring_starts (bool): Whether to use exploring starts for initial state-action pairs.
            episodes (int): Number of episodes for training.
            max_steps_per_episode (int): Maximum steps per episode.
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.exploring_starts = exploring_starts
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Action-value function
        self.Returns = defaultdict(list)  # Returns for each state-action pair
        self.policy = {s: np.random.choice(self.env.action_space.n) for s in range(self.env.observation_space.n)}  # Random initial policy
        
        # Track history for convergence analysis
        self.V_history = []
        self.Q_history = []
        self.policy_history = []

    def generate_episode(self):
        """
        Generates an episode by following the current policy until terminal state or maximum steps.

        Returns:
            list: A list of (state, action, reward) tuples representing the episode.
        """
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
        """
        Selects an action based on epsilon-greedy policy.

        Args:
            state (int): Current state.

        Returns:
            int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def run(self):
        """
        Runs the Monte Carlo learning algorithm for a specified number of episodes.

        Returns:
            dict: Optimal policy derived from Q values.
        """
        for episode_num in range(1, self.episodes + 1):
            # Generate an episode with or without exploring starts
            if self.exploring_starts:
                state = self.env.observation_space.sample()
                action = self.env.action_space.sample()
                self.env.state = self.index_to_state(state)
                episode = self.generate_episode_with_start(state, action)
            else:
                episode = self.generate_episode()

            G = 0  # Initialize return
            visited_state_action_pairs = set()

            # Process the episode in reverse to calculate returns
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                # Update Q values if this is the first visit to the state-action pair
                if (state, action) not in visited_state_action_pairs:
                    self.Returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.Returns[(state, action)])
                    self.policy[state] = np.argmax(self.Q[state])
                    visited_state_action_pairs.add((state, action))

            # Record convergence data for analysis
            self.record_convergence()

            if episode_num % 100 == 0:
                print(f"Episode {episode_num}/{self.episodes} completed.")

        return self.policy

    def generate_episode_with_start(self, start_state, start_action):
        """
        Generates an episode beginning with a specified state-action pair.

        Args:
            start_state (int): Initial state.
            start_action (int): Initial action.

        Returns:
            list: A list of (state, action, reward) tuples representing the episode.
        """
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

    def record_convergence(self):
        """
        Records the current value function, action-value function, and policy for convergence analysis.
        """
        # Compute V(s) as the maximum Q(s, a)
        V = np.zeros(self.env.observation_space.n)
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        policy_array = np.zeros(self.env.observation_space.n, dtype=int)

        for s in range(self.env.observation_space.n):
            Q_s = self.Q[s]
            V[s] = np.max(Q_s)
            Q[s, :] = Q_s
            policy_array[s] = self.policy[s]

        self.V_history.append(V)
        self.Q_history.append(Q)
        self.policy_history.append(policy_array)

    def index_to_state(self, index):
        """
        Converts an index to a state format compatible with the environment.

        Args:
            index (int): Flattened index.

        Returns:
            int or tuple: Converted state.
        """
        if hasattr(self.env, 'index_to_state'):
            return self.env.index_to_state(index)
        else:
            return index  # For environments where the state is an index


# TDSolver class
class TDSolver:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=1000, max_steps_per_episode=100, method="sarsa"):
        """
        Initializes the TDSolver with parameters for discounting, learning rate, exploration,
        episode settings, and choice of TD method.

        Args:
            env (gym.Env): The environment to solve.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for TD updates.
            epsilon (float): Probability for epsilon-greedy action selection.
            episodes (int): Number of episodes for training.
            max_steps_per_episode (int): Maximum steps per episode.
            method (str): TD method to use ("sarsa" or "q_learning").
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.method = method  # "sarsa" or "q_learning"
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Action-value function
        self.policy = {}

        # Track history for convergence analysis
        self.V_history = []
        self.Q_history = []
        self.policy_history = []
        self.episode_rewards = []  # Track rewards per episode

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy based on current Q-values.

        Args:
            state (int): Current state.

        Returns:
            int: Selected action.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def run(self):
        """
        Runs the TD learning algorithm (SARSA or Q-learning) for a specified number of episodes.

        Returns:
            dict: Derived optimal policy based on Q values.
        """
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            action = self.select_action(state)
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < self.max_steps_per_episode:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.select_action(next_state) if self.method == "sarsa" else np.argmax(self.Q[next_state])

                # Calculate TD target based on method
                if self.method == "sarsa":
                    td_target = reward + self.gamma * self.Q[next_state][next_action]
                else:  # Q-learning
                    td_target = reward + self.gamma * np.max(self.Q[next_state])

                td_error = td_target - self.Q[state][action]

                # Update Q-value with TD error
                self.Q[state][action] += self.alpha * td_error

                # Update policy to be greedy with respect to current Q-values
                self.policy[state] = np.argmax(self.Q[state])

                # Move to the next state and action
                state = next_state
                action = next_action if self.method == "sarsa" else self.select_action(next_state)
                steps += 1
                total_reward += reward

            # Record convergence data for analysis
            self.record_convergence()
            self.episode_rewards.append(total_reward)  # Track rewards for performance analysis

            # Debug print
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.episodes} completed.")

        return self.policy

    def record_convergence(self):
        """
        Records the current value function, action-value function, and policy for convergence analysis.
        """
        # Compute V(s) as the maximum Q(s, a)
        V = np.zeros(self.env.observation_space.n)
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        policy_array = np.zeros(self.env.observation_space.n, dtype=int)

        for s in range(self.env.observation_space.n):
            Q_s = self.Q[s]
            V[s] = np.max(Q_s)
            Q[s, :] = Q_s
            policy_array[s] = self.policy.get(s, 0)  # Default to 0 if state not in policy

        # Append current values for convergence tracking
        self.V_history.append(V)
        self.Q_history.append(Q)
        self.policy_history.append(policy_array)
