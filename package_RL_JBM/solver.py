import gym
from gym import spaces
import numpy as np
import pygame

class BoatSolver:
    def __init__(self, env, transition_probs, reward_probs, gamma=0.99):
        self.env = env
        self.transition_probs = transition_probs
        self.reward_probs = reward_probs
        self.gamma = gamma
        self.values = {s: 0 for s in range(env.observation_space.n)}
        self.policy = {s: np.random.choice([0, 1]) for s in range(env.observation_space.n)}

    def policy_iteration(self, threshold=0.001):
        is_policy_stable = False
        while not is_policy_stable:
            self.policy_evaluation()
            is_policy_stable = self.policy_improvement()

    def policy_evaluation(self, threshold=0.001):
        while True:
            delta = 0
            for s in self.values:
                v = self.values[s]
                self.values[s] = sum(self.transition_probs.get((s, self.policy[s], s_prime), 0) *
                                     (r + self.gamma * self.values[s_prime])
                                     for s_prime in range(self.env.observation_space.n) 
                                     for r in range(-10, 11)  # Assuming reward range is -10 to 10
                                     if (s, self.policy[s], s_prime, r) in self.reward_probs)
                delta = max(delta, abs(v - self.values[s]))
            if delta < threshold:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.env.observation_space.n):
            old_action = self.policy[s]
            self.policy[s] = np.argmax([sum(self.transition_probs.get((s, a, s_prime), 0) *
                                            (r + self.gamma * self.values[s_prime])
                                            for s_prime in range(self.env.observation_space.n)
                                            for r in range(-10, 10)  # Assuming reward range is -10 to 10
                                            if (s, a, s_prime, r) in self.reward_probs)
                                        for a in range(self.env.action_space.n)])
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def value_iteration(self, threshold=0.001):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.values[s]
                self.values[s] = max(sum(self.transition_probs.get((s, a, s_prime), 0) *
                                         (r + self.gamma * self.values[s_prime])
                                         for s_prime in range(self.env.observation_space.n)
                                         for r in range(-10, 10)  # Assuming reward range is -10 to 10
                                         if (s, a, s_prime, r) in self.reward_probs)
                                     for a in range(self.env.action_space.n))
                delta = max(delta, abs(v - self.values[s]))
            if delta < threshold:
                break

    def solve(self, method='policy_iteration'):
        if method == 'policy_iteration':
            self.policy_iteration()
        elif method == 'value_iteration':
            self.value_iteration()
        else:
            raise ValueError("Invalid method chosen: use 'policy_iteration' or 'value_iteration'")
