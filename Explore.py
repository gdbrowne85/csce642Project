# This script trains the agent to explore by rewarding it for increasing the distance from the starting position.

# This script trains the agent to explore based on distance change from origin and prevents it from getting stuck.

import vizdoom as vzd
from Simpler_CDQN import CDQN
from gym.spaces import Box
import numpy as np
from PIL import Image
from collections import deque
from gym.spaces import Discrete
import math


def process_screen_buffer(screen_buffer, width, height):
    """
    Converts the screen buffer to grayscale and resizes it to the specified dimensions.
    """
    if screen_buffer is None or len(screen_buffer) == 0:
        raise ValueError("Screen buffer is empty or None!")

    image = Image.fromarray(screen_buffer.astype('uint8'), mode='RGB')
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.float32)  # Convert back to NumPy array


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
    epsilon=0.3,
    batch_size=32,
    update_target_estimator_every=1000
)


class DoomEnv:
    """
    Custom wrapper for the ViZDoom environment.
    """
    def __init__(self, game, width=80, height=60, frame_stack=4, max_steps=float('inf'), allowed_actions=None):
        self.width = width
        self.height = height
        self.game = game
        self.frame_stack = frame_stack
        self.max_steps = max_steps
        self.steps_taken = 0
        self.starting_position = None
        self.episode_done = False

        # Variables for calculating rewards
        self.previous_distance = 0
        self.stuck_counter = 0
        self.stuck_threshold = 10  # Threshold for determining if stuck
        self.position_history = deque(maxlen=self.stuck_threshold)

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

        # Use allowed actions to determine active actions
        if allowed_actions is not None:
            self.allowed_actions = allowed_actions
        else:
            self.allowed_actions = list(range(len(self.all_actions)))

        self.actions = [self.all_actions[i] for i in self.allowed_actions]
        self.action_space = Discrete(len(self.actions))

        # Observation space for stacked grayscale frames
        screen_shape = (frame_stack, height, width)
        self.observation_space = Box(low=0, high=255, shape=screen_shape, dtype=np.float32)

        # Frame buffer for stacking frames
        self.frame_buffer = deque(maxlen=frame_stack)

    def _stack_frames(self, frame):
        if len(self.frame_buffer) == 0:
            # Initialize the frame buffer with the same frame
            for _ in range(self.frame_stack):
                self.frame_buffer.append(frame)
        else:
            self.frame_buffer.append(frame)

        # Stack frames along the channel dimension
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames

    def reset(self):
        self.game.new_episode()
        self.steps_taken = 0
        self.episode_done = False
        self.stuck_counter = 0
        self.position_history.clear()

        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            raise ValueError("Game state or screen buffer is empty!")

        # Process screen buffer
        screen_buffer = state.screen_buffer
        processed_screen_buffer = process_screen_buffer(screen_buffer, self.width, self.height)

        # Initialize the frame buffer and stack frames
        stacked_frames = self._stack_frames(processed_screen_buffer)

        # Store initial position
        self.starting_position = [
            self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
        ]
        self.previous_distance = 0

        return stacked_frames, {}

    def step(self, action_index):
        action = self.actions[action_index]
        self.game.make_action(action)
        done = self.game.is_episode_finished()
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.episode_done = True

        reward = 0
        if not (done or self.episode_done):
            state = self.game.get_state()
            if state is None or state.screen_buffer is None:
                raise ValueError("Game state or screen buffer is empty!")

            # Process screen buffer
            screen_buffer = state.screen_buffer
            processed_screen_buffer = process_screen_buffer(screen_buffer, self.width, self.height)

            # Stack frames
            stacked_frames = self._stack_frames(processed_screen_buffer)

            # Get current position
            current_position = [
                self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
                self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
            ]

            # Calculate rewards
            current_distance = self._calculate_distance(self.starting_position, current_position)
            distance_reward = current_distance - self.previous_distance

            # Penalize no progress or being stuck
            self.position_history.append(current_position)
            if len(self.position_history) == self.stuck_threshold and len(set(map(tuple, self.position_history))) == 1:
                reward -= 5  # Stuck penalty
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            # Encourage movement and exploration
            reward += distance_reward + 0.01  # Small action reward

            # Update previous distance
            self.previous_distance = current_distance

            return stacked_frames, reward, done, {}, {}
        else:
            # Handle terminal state
            empty_screen = np.zeros((self.frame_stack, self.height, self.width), dtype=np.float32)
            return empty_screen, reward, done, {}, {}

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


if __name__ == "__main__":
    width = 80
    height = 60
    game = vzd.DoomGame()
    #level_path = ("my_way_home.cfg")
    #game.load_config(level_path)

    game.clear_available_buttons()
    buttons = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_FORWARD, vzd.Button.ATTACK,
               vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.JUMP, vzd.Button.USE]
    for button in buttons:
        game.add_available_button(button)

    game.clear_available_game_variables()
    game_variables = [vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.POSITION_Z]
    for game_variable in game_variables:
        game.add_available_game_variable(game_variable)

    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    # Specify allowed actions (exclude ATTACK)
    allowed_actions = [0, 1, 2, 4, 5, 7]  # MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, USE

    # Initialize the environment
    env = DoomEnv(game, width=width, height=height, max_steps=1199, allowed_actions=allowed_actions)

    # Initialize the DQN agent
    dqn_agent = CDQN(
        env=env,
        eval_env=env,
        options=options,
        num_actions=len(env.all_actions),  # Keep full action space length
        in_channels=env.frame_stack  # Only screen buffer is passed now
    )

    # Epsilon decay parameters
    epsilon_decay_rate = 0.997
    epsilon_min = 0.01

    num_episodes = 400
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not (done or env.episode_done):
            probabilities = dqn_agent.epsilon_greedy(state)
            action = np.random.choice(env.action_space.n, p=probabilities)

            next_state, reward, done, _, _ = env.step(action)
            dqn_agent.memorize(state, action, reward, next_state, done)
            dqn_agent.replay()

            total_reward += reward
            state = next_state
            episode_steps += 1
            dqn_agent.n_steps += 1

            if dqn_agent.n_steps % dqn_agent.options.update_target_estimator_every == 0:
                dqn_agent.update_target_model()

        # Decay epsilon after each episode
        dqn_agent.options.epsilon = max(dqn_agent.options.epsilon * epsilon_decay_rate, epsilon_min)

        print(f"Episode {episode + 1} finished with total reward: {total_reward}, steps: {episode_steps}, "
              f"epsilon: {dqn_agent.options.epsilon:.4f}")
        dqn_agent.episode_rewards.append(total_reward)
        dqn_agent.plot_rewards(save_path='exploration.png')
        model_save_path = "explore.pth"
        dqn_agent.save_model(model_save_path)
    env.close()