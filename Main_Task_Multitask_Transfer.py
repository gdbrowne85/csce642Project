# This script trains a selector network to choose between four pretrained subtask models and applies a custom reward function.

import vizdoom as vzd
from Simpler_CDQN import CDQN
from gym.spaces import Box
import numpy as np
from PIL import Image
from collections import deque
from gym.spaces import Discrete
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math


def process_screen_buffer(screen_buffer, width, height):
    """
    Converts the screen buffer to grayscale and resizes it to the specified dimensions.
    """
    image = Image.fromarray(screen_buffer.astype('uint8'), mode='RGB')
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.float32)


class Options:
    """
    Class to store training options.
    """
    def __init__(self, replay_memory_size, layers, alpha, gamma, epsilon, batch_size, update_target_estimator_every):
        self.replay_memory_size = replay_memory_size
        self.layers = layers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every


options = Options(
    replay_memory_size=100000,
    layers=[128, 128],
    alpha=0.001,
    gamma=0.9,
    epsilon=0.05,
    batch_size=32,
    update_target_estimator_every=1000
)


class DoomEnv:
    """
    Custom wrapper for the ViZDoom environment.
    """
    def __init__(self, game, width=80, height=60, frame_stack=4, max_steps=1000, allowed_actions=None):
        self.width = width
        self.height = height
        self.game = game
        self.frame_stack = frame_stack
        self.max_steps = max_steps
        self.steps_taken = 0
        self.episode_done = False

        # Reward function components
        self.previous_distance = 0
        self.previous_hitcount = 0
        self.previous_ammo = 0
        self.previous_health = 0
        self.previous_itemcount = 0

        # Define all possible actions
        self.all_actions = [
            [True, False, False, False, False, False, False, False],  # MOVE_LEFT
            [False, True, False, False, False, False, False, False],  # MOVE_RIGHT
            [False, False, True, False, False, False, False, False],  # MOVE_FORWARD
            [False, False, False, True, False, False, False, False],  # ATTACK
            [False, False, False, False, True, False, False, False],  # TURN_LEFT
            [False, False, False, False, False, True, False, False],  # TURN_RIGHT
            [False, False, False, False, False, False, True, False],  # JUMP
            [False, False, False, False, False, False, False, True],  # USE
        ]

        # Mask actions dynamically if allowed_actions is provided
        if allowed_actions is not None:
            self.allowed_actions = allowed_actions
        else:
            self.allowed_actions = list(range(len(self.all_actions)))

        self.actions = [self.all_actions[i] for i in self.allowed_actions]
        self.action_space = Discrete(len(self.actions))

        # Observation space
        channels = self.frame_stack
        screen_shape = (channels, height, width)
        self.observation_space = Box(low=0, high=255, shape=screen_shape, dtype=np.float32)

        self.frame_buffer = deque(maxlen=frame_stack)

    def _stack_frames(self, frame):
        if len(self.frame_buffer) == 0:
            for _ in range(self.frame_stack):
                self.frame_buffer.append(frame)
        else:
            self.frame_buffer.append(frame)
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames

    def reset(self):
        self.game.new_episode()
        self.steps_taken = 0
        self.episode_done = False

        state = self.game.get_state()
        screen_buffer = np.array(state.screen_buffer, dtype=np.float32)
        processed_screen_buffer = process_screen_buffer(screen_buffer, self.width, self.height)

        # Initialize variables for reward calculation
        self.previous_distance = 0
        self.previous_hitcount = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        self.previous_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        self.previous_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self.previous_itemcount = self.game.get_game_variable(vzd.GameVariable.ITEMCOUNT)

        return self._stack_frames(processed_screen_buffer), {}

    def step(self, action_index):
        action = self.actions[action_index]
        self.game.make_action(action)
        done = self.game.is_episode_finished()
        self.steps_taken += 1

        if done or self.steps_taken >= self.max_steps:
            self.episode_done = True
            return np.zeros_like(self.frame_buffer), 0, done, {}, {}

        # Calculate reward
        state = self.game.get_state()
        screen_buffer = np.array(state.screen_buffer, dtype=np.float32)
        processed_screen_buffer = process_screen_buffer(screen_buffer, self.width, self.height)

        current_position = [
            self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
        ]
        current_hitcount = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        current_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        current_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        current_itemcount = self.game.get_game_variable(vzd.GameVariable.ITEMCOUNT)

        new_distance = self._calculate_distance((0, 0, 0), current_position)
        distance_reward = 0.01 * (new_distance - self.previous_distance)
        hits_reward = current_hitcount - self.previous_hitcount
        items_reward = 10 * (current_itemcount - self.previous_itemcount)
        health_penalty = self.previous_health - current_health
        ammo_penalty = 0.1 * (self.previous_ammo - current_ammo)

        reward = distance_reward + hits_reward + items_reward - health_penalty - ammo_penalty

        # Update previous values
        self.previous_distance = new_distance
        self.previous_hitcount = current_hitcount
        self.previous_ammo = current_ammo
        self.previous_health = current_health
        self.previous_itemcount = current_itemcount

        return self._stack_frames(processed_screen_buffer), reward, done, {}, {}

    def _calculate_distance(self, start, end):
        """
        Calculate Euclidean distance between two 3D points.
        """
        return math.sqrt(
            (end[0] - start[0]) ** 2 +
            (end[1] - start[1]) ** 2 +
            (end[2] - start[2]) ** 2
        )

    def close(self):
        self.game.close()


class SelectorNetwork(nn.Module):
    """
    Network to select the best subtask model based on the state.
    """

    def __init__(self, input_height, input_width, frame_stack, num_models):
        super(SelectorNetwork, self).__init__()

        # Calculate input dimensions after flattening
        self.input_dim = input_height * input_width * frame_stack

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # Adjust input dimension dynamically
            nn.ReLU(),
            nn.Linear(128, num_models)
        )

    def forward(self, state):
        # Ensure state is flattened
        state = state.view(state.size(0), -1)  # Flatten state [batch_size, input_dim]
        return self.fc(state)


if __name__ == "__main__":
    # Initialize VizDoom
    game = vzd.DoomGame()
    #game.load_config("defend_the_center.cfg")
    #game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(True)

    # Configure allowed buttons
    buttons = [
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.ATTACK,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.JUMP,
        vzd.Button.USE,
    ]

    for button in buttons:
        game.add_available_button(button)

    # Configure game variables
    game.add_available_game_variable(vzd.GameVariable.HITCOUNT)
    game.add_available_game_variable(vzd.GameVariable.AMMO2)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)

    # Game setup
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    # Allowed actions setup
    allowed_actions = [0, 1, 2, 3, 4, 5, 6, 7]  # All actions are allowed
    env = DoomEnv(game, width=80, height=60, max_steps=float('inf'), allowed_actions=allowed_actions)

    state_dim = env.observation_space.shape[1] * env.observation_space.shape[2]
    num_models = 4
    num_actions = len(env.actions)

    # Load pretrained subtask models
    subtask_models = []
    model_paths = ['high_accuracy.pth', 'item_gathering.pth', 'dodge.pth', 'explore.pth']
    for model_path in model_paths:
        model = CDQN(env, env, options, num_actions)
        model.load_model(model_path)
        subtask_models.append(model)

    # Calculate dimensions dynamically from environment
    input_height, input_width = env.height, env.width
    frame_stack = env.frame_stack

    # Initialize selector network
    selector_network = SelectorNetwork(
        input_height=input_height,
        input_width=input_width,
        frame_stack=frame_stack,
        num_models=num_models
    ).cuda()

    selector_optimizer = optim.Adam(selector_network.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train selector network
    num_episodes = 1000
    replay_memory = deque(maxlen=10000)
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_flattened = torch.FloatTensor(state.flatten()).unsqueeze(0).cuda()
        total_reward = 0
        done = False

        while not done:
            # Debugging: Log episode and step count
            print(f"Episode: {episode + 1}, Step: {env.steps_taken}")

            # Epsilon-greedy selection of the subtask model
            if random.random() < epsilon:
                selected_model = random.randint(0, num_models - 1)
            else:
                with torch.no_grad():
                    selector_output = selector_network(state_flattened)
                    selected_model = torch.argmax(selector_output).item()
            # Select action using the chosen subtask model
            probs = subtask_models[selected_model].epsilon_greedy(state)
            action = np.random.choice(8, p=probs)

            # Debugging: Log selected model and action
            print(f"Selected Model: {selected_model}, Action: {action}")

            # Step in the environment
            next_state, reward, done, _, _ = env.step(action)
            next_state_flattened = torch.FloatTensor(next_state.flatten()).unsqueeze(0).cuda()

            # Penalize inactivity
            if reward == 0:
                reward -= 0.1

            # Store transition in replay memory
            replay_memory.append((state_flattened, selected_model, reward, next_state_flattened, done))
            total_reward += reward

            # Update state
            state_flattened = next_state_flattened

            # Train selector network if replay memory is sufficient
            if len(replay_memory) > 64:
                minibatch = random.sample(replay_memory, 64)
                states, model_indices, rewards, next_states, dones = zip(*minibatch)

                states = torch.cat(states)
                model_indices = torch.LongTensor(model_indices).cuda()
                rewards = torch.FloatTensor(rewards).cuda()

                selector_output = selector_network(states)
                loss = loss_fn(selector_output, model_indices)

                selector_optimizer.zero_grad()
                loss.backward()
                selector_optimizer.step()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Debugging: Log episode results
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    env.close()

