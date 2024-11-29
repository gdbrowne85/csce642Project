import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from Abstract_Solver import AbstractSolver, Statistics

import matplotlib
import matplotlib
matplotlib.use("TkAgg")  # Or "Agg" for headless systems

import cv2
import numpy as np


class QFunction(nn.Module):
    def __init__(self, num_actions, in_channels=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Halves height and width
        )

        # Calculate output size of the convolutional layers dynamically
        conv_output_height = 60 // 2  # After MaxPool2d
        conv_output_width = 80 // 2
        conv_out_size = 32 * conv_output_height * conv_output_width

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # Dynamically adjust input size
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, screen_buffer):
        # Pass through convolutional layers
        conv_output = self.conv_layers(screen_buffer)
        # Flatten the convolutional output
        flattened = torch.flatten(conv_output, start_dim=1)
        # Pass through fully connected layers
        return self.fc_layers(flattened)

class CDQN(AbstractSolver):
    def __init__(self, env, eval_env, options, num_actions, in_channels=4):  # Grayscale: 4 stacked frames
        # GPU/CPU device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_rewards = []
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)

        # Initialize the model with only grayscale screen buffer input
        self.model = QFunction(num_actions, in_channels).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        self.loss_fn = nn.SmoothL1Loss()

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.replay_memory = deque(maxlen=self.options.replay_memory_size)
        self.n_steps = 0

    def save_model(self, save_path):
        """
        Save the trained model's state dictionary to the specified path.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load a pretrained model's state dictionary from the specified path.
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)
        self.target_model = deepcopy(self.model)  # Update the target model as well
        print(f"Model loaded from {load_path}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Target network updated.")

    def epsilon_greedy(self, state):
        """
        Selects an action using the epsilon-greedy policy.
        Only allows actions in `allowed_actions`.
        """
        # Ensure state is correctly processed as a tensor
        screen_buffer = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dim

        # Get Q-values
        q_values = self.model(screen_buffer)

        # Initialize probabilities array (for all actions)
        epsilon = self.options.epsilon
        prob_vector = np.zeros(len(self.env.all_actions))  # Zero probabilities for all actions

        # Set exploration probability for allowed actions
        for action in self.env.allowed_actions:
            prob_vector[action] = epsilon / len(self.env.allowed_actions)

        # Mask invalid actions for exploitation
        q_values_masked = q_values.clone()
        for action in range(q_values.size(1)):
            if action not in self.env.allowed_actions:
                q_values_masked[:, action] = float('-inf')

        # Get the best valid action (max Q-value among allowed actions)
        best_action = torch.argmax(q_values_masked).item()
        prob_vector[best_action] += (1 - epsilon)  # Add exploitation probability to the best action

        # Filter probabilities to match the size of allowed actions
        filtered_probs = np.array([prob_vector[action] for action in self.env.allowed_actions])

        # Normalize the probabilities to sum to 1
        filtered_probs /= filtered_probs.sum()

        return filtered_probs

    def compute_target_values(self, next_states, rewards, dones):
        next_screen_buffers = torch.as_tensor(
            np.stack([buf.cpu().numpy() for buf in next_states]),  # Fix: Move tensors to CPU before converting to NumPy
            dtype=torch.float32, device=self.device
        )

        # Add channel dimension if necessary
        if len(next_screen_buffers.shape) == 3:  # [batch_size, H, W]
            next_screen_buffers = next_screen_buffers.unsqueeze(1)

        # Get the action with the highest Q-value from the online model
        next_actions = self.model(next_screen_buffers).argmax(dim=1)

        # Use the target model to calculate the Q-value of these actions
        next_q_values = self.target_model(next_screen_buffers)
        next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        dones = dones.to(self.device).float()
        target_q = rewards.to(self.device) + self.options.gamma * next_q_values * (1 - dones)
        return target_q

    def replay(self):
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            screen_buffers = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
            next_screen_buffers = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

            # Compute current Q-values
            q_values = self.model(screen_buffers)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            with torch.no_grad():
                target_q = self.compute_target_values(next_screen_buffers, rewards, dones)

            # Compute loss
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        state = self.env.reset()
        total_reward = 0
        episode_steps = 0

        for _ in range(self.options.steps):
            self.n_steps += 1
            probabilities = self.epsilon_greedy(state)
            action = np.random.choice(self.env.action_space.n, p=probabilities)

            next_state, reward, done, _ = self.env.step(action)

            self.memorize(state, action, reward, next_state, done)
            self.replay()

            total_reward += reward
            episode_steps += 1
            state = next_state

            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()

            if done:
                self.statistics[Statistics.Rewards.value] = total_reward
                self.statistics[Statistics.Steps.value] = episode_steps
                break

    def __str__(self):
        return "DQN"


    def plot_rewards(self, smoothing_window=10, save_path=None):
        plt.ion()  # Ensure interactive mode is on for dynamic updates
        plt.clf()  # Clear the figure

        # Plot raw episodic rewards
        plt.plot(self.episode_rewards, label="Episodic Reward")

        # Smooth rewards if there are enough episodes
        if len(self.episode_rewards) >= smoothing_window:
            smoothed_rewards = np.convolve(
                self.episode_rewards, np.ones(smoothing_window) / smoothing_window, mode='valid'
            )
            plt.plot(
                range(smoothing_window - 1, len(self.episode_rewards)),
                smoothed_rewards,
                label="Smoothed Reward",
                color="orange",
            )

        # Configure the plot
        plt.xlabel("Episode Number")
        plt.ylabel("Total Reward")
        plt.title("Episodic Reward vs. Episode Number")
        plt.legend()
        plt.grid()

        # Save the plot if save_path is specified
        if save_path:
            plt.savefig(save_path)  # Save the current plot

        # Pause to allow the plot to render during training
        plt.pause(0.01)

    def create_greedy_policy(self):
        def policy_fn(state):
            screen_buffer = state
            screen_buffer_tensor = torch.as_tensor(screen_buffer, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(screen_buffer_tensor)
            return torch.argmax(q_values).item()

        return policy_fn

    def choose_action(self, state):
        """
        Computes the Q-values for a given state and selects the greedy action.

        Args:
            state (numpy.ndarray): The current state as a NumPy array.

        Returns:
            tuple: A tuple containing:
                - greedy_action (int): The index of the greedy action.
        """
        # Ensure state is correctly processed as a tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dim

        # Pass the state through the model to compute Q-values
        q_values = self.model(state_tensor)
        print ('Q values: ', q_values)
        # Select the greedy action (action with the highest Q-value)
        greedy_action = torch.argmax(q_values, dim=1).item()

        return greedy_action

