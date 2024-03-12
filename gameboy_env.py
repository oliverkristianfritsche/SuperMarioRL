import gymnasium as gym
from pyboy import PyBoy, WindowEvent
from gym import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import FrameStorage, bcd_to_int
import cv2
import gymnasium as gym
from pyboy import PyBoy, WindowEvent
from gym import spaces
import numpy as np
import os
import cv2

class GameBoyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, rom_path, observation_shape=None, game_buttons=[], window_type="headless"):
        super(GameBoyEnv, self).__init__()
        assert os.path.exists(rom_path), f"ROM {rom_path} not found"
        self.pyboy = PyBoy(rom_path, window_type=window_type)
        self.game_buttons = game_buttons
        self.action_space = spaces.Discrete(len(self.game_buttons))
        self.observation_shape = observation_shape if observation_shape else (160, 144)  # Default Game Boy screen size
        self.observation_space = spaces.Box(low=0, high=255, shape=(*self.observation_shape, 3), dtype=np.uint8)
        self.pyboy.set_emulation_speed(0)  # Run as fast as possible

    def step(self, action):
        self.pyboy.send_input(self.game_buttons[action]['press'])
        for _ in range(self.n_ticks_per_step - 1):
            self.pyboy.tick()
        self.pyboy.send_input(self.game_buttons[action]['release'])
        self.pyboy.tick()  # One final tick for the action to take effect
        
        observation = self._get_observation()
        done = self._check_game_over()
        reward, info = self._calculate_reward(observation, done)
        return observation, reward, done, info

    def reset(self):
        self.pyboy.reset_game()
        return self._get_observation()

    def render(self, mode='human'):
        img = self.pyboy.botsupport_manager().screen().screen_ndarray()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow("Game", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        self.pyboy.stop()
        cv2.destroyAllWindows()

    def _get_observation(self):
        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
        return cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
    
    def _check_game_over(self):
        # General method; should be overridden by subclass
        return False

    def _calculate_reward(self, observation, done):
        # General method; should be overridden by subclass
        return 0, {}



class MarioGameBoyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    n_ticks_per_step = 3
    max_steps = 10000
    death_scalar = 15
    survive_scalar = 0
    frame_scalar = 0.000
    level_scalar = 0.0
    coin_scalar = 1.00
    score_scalar = .01
    go_right_scalar = 0.05
    go_left_scalar = -0.01
    reward_scalar=0.01

    prev_level_reward = 0
    prev_coin_reward = 0
    prev_score_reward = 0

    # Define Mario-specific Game Boy buttons
    gameboy_buttons = [
        # [WindowEvent.PRESS_BUTTON_A],
        # [WindowEvent.PRESS_BUTTON_B],
        # [WindowEvent.PRESS_ARROW_RIGHT],
        # [WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_A],
        [WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_B],
        [WindowEvent.PRESS_ARROW_RIGHT,WindowEvent.PRESS_BUTTON_A,WindowEvent.PRESS_BUTTON_B,],
    
        [WindowEvent.PRESS_ARROW_LEFT],
        # WindowEvent.PRESS_ARROW_UP,
        # WindowEvent.PRESS_ARROW_DOWN,
        # WindowEvent.PRESS_BUTTON_START,
        # WindowEvent.PRESS_BUTTON_SELECT
    ]
    gameboy_buttons_release = [
        # [WindowEvent.RELEASE_BUTTON_A],
        # [WindowEvent.RELEASE_BUTTON_B],
        # [WindowEvent.RELEASE_ARROW_RIGHT],
        # [WindowEvent.RELEASE_ARROW_RIGHT,WindowEvent.RELEASE_BUTTON_A],
        [WindowEvent.RELEASE_ARROW_RIGHT,WindowEvent.RELEASE_BUTTON_B],
        [WindowEvent.RELEASE_ARROW_RIGHT,WindowEvent.RELEASE_BUTTON_A,WindowEvent.RELEASE_BUTTON_B],
        [WindowEvent.RELEASE_ARROW_LEFT],
        # WindowEvent.RELEASE_ARROW_UP,
        # WindowEvent.RELEASE_ARROW_DOWN,
        # WindowEvent.RELEASE_BUTTON_START,
        # WindowEvent.RELEASE_BUTTON_SELECT
    ]

    def __init__(self, rom_path, observation_shape=None,gameboy_buttons=None,gameboy_buttons_release=None, window_type="headless"):
        assert os.path.exists(rom_path), f"ROM {rom_path} not found"
        self.gameboy_buttons = gameboy_buttons if gameboy_buttons else self.gameboy_buttons
        self.gameboy_buttons_release = gameboy_buttons_release if gameboy_buttons_release else self.gameboy_buttons_release
        self.pyboy = PyBoy(rom_path)
        self.initialize_game()
        self.action_space = spaces.Discrete(len(self.gameboy_buttons))
        self.observation_shape = observation_shape if observation_shape else self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(*self.observation_shape, 3), dtype=np.uint8)
        self.unique_frames = FrameStorage()
        print(f"Observation shape: {self.observation_shape}")
        self.last_action = None

    def initialize_game(self):
        self.pyboy.set_emulation_speed(0)
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        for _ in range(100):
            self.pyboy.tick()

        state_path = "roms/initial_state_mario.state"
        if not os.path.exists(state_path):
            self.pyboy.save_state(open(state_path, "wb"))

    def step(self, action):
        if self.last_action is not None and action != self.last_action:
            for i in range(len(self.gameboy_buttons_release[self.last_action])):
                self.pyboy.send_input(self.gameboy_buttons_release[self.last_action][i])
        
        for i in range(len(self.gameboy_buttons[action])):
            self.pyboy.send_input(self.gameboy_buttons[action][i])
        
        for _ in range(self.n_ticks_per_step):
            self.time_steps += 1
            self.pyboy.tick()
        done = self.check_game_over()
        reward, info = self.calculate_reward(done)
        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
        
        #crop top 32 pixels
        observation = observation[32:,:,:].copy()
        observation = cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
   
        self.last_action = action

        return observation, reward, done, info
    
    def calculate_reward(self,done):

        instant_velocity_abs = min(self.pyboy.get_memory_value(0xC20C),self.death_scalar)
        direction = self.pyboy.get_memory_value(0xC20D)
        instant_velocity= instant_velocity_abs
        if direction == 32:
            instant_velocity *= -1
        elif direction == 0:
            instant_velocity = 0

        reward = instant_velocity - self.n_ticks_per_step #penalize for not moving
        if done:
            reward -= self.death_scalar

        # print(reward,instant_velocity,direction)

        #clip reward [-15,15]
        reward = max(-15,min(15,reward))

        return reward*self.reward_scalar, {}
        

    def calculate_reward_old(self, observation, done):
        current_level_reward = self.get_level_reward()
        current_coin_reward = self.get_coin_reward()
        current_score_reward = self.get_score_reward()

        delta_level_reward = current_level_reward - self.prev_level_reward
        delta_coin_reward = current_coin_reward - self.prev_coin_reward
        delta_score_reward = current_score_reward - self.prev_score_reward

        delta_coin_reward = max(delta_coin_reward, 0)
        delta_level_reward = max(delta_level_reward, 0)
        delta_score_reward = max(delta_score_reward, 0)

        frame_reward = self.unique_frames.add_frame(observation) * self.frame_scalar

        reward = frame_reward + delta_level_reward * self.level_scalar + delta_coin_reward * self.coin_scalar + delta_score_reward * self.score_scalar

        self.prev_level_reward = current_level_reward
        self.prev_coin_reward = current_coin_reward
        self.prev_score_reward = current_score_reward

        return reward, {
            "level_reward": delta_level_reward, 
            "coin_reward": delta_coin_reward, 
            "score_reward": delta_score_reward
        }

    def get_frame_reward(self, frame_reward):
        return frame_reward * self.frame_scalar
    
    def get_level_reward(self):
        current_world = self.pyboy.get_memory_value(0x982C)-1
        current_stage = self.pyboy.get_memory_value(0x982E)-1
        level = (current_world * 10) + current_stage  # Simplistic level calculation
        return level * self.level_scalar  # Adjust the reward scaling as needed

    def get_coin_reward(self):
        # Read the total amount of coins
        coins = self.pyboy.get_memory_value(0xFFFA)
        return coins * self.coin_scalar  # Adjust the reward scaling as needed
    
    def get_score_reward(self):
        return (self.get_score()-(100*self.pyboy.get_memory_value(0xFFFA))) * self.score_scalar  # Adjust the reward scaling as needed (self.get_score()-(100*self.pyboy.get_memory_value(0xFFFA))) * self.score_scalar 

    
    def get_score(self):
        score_bytes = [self.pyboy.get_memory_value(addr) for addr in (0xC0A0, 0xC0A1, 0xC0A2)]
        score = bcd_to_int(score_bytes)
        return score
    
    def check_game_over(self):
        return self.pyboy.get_memory_value(0xDA15) < 2 or self.time_steps > self.max_steps

    def reset(self):
        self.pyboy.load_state(open("roms/initial_state_mario.state", "rb"))
        self.unique_frames = FrameStorage()
        self.time_steps = 0
        self.prev_level_reward = 0
        self.prev_coin_reward = 0
        self.prev_score_reward = 0

        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
        observation = cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
        return observation

    def close(self):
        self.pyboy.stop()

    def render(self, mode='human'):
        img = self.pyboy.botsupport_manager().screen().screen_ndarray()
        if mode == 'human':
            plt.imshow(img)
            plt.show()
        elif mode == 'rgb_array':
            return img
        

class PacManGameBoyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    n_ticks_per_step = 3
    max_steps = 10000
    time_steps = 0

    # Adjusting scalar values for Pac-Man
    death_scalar = -5
    pellet_scalar = 1.0
    fruit_scalar = 5.0
    ghost_scalar = 10.0
    score_scalar = 0.01
    level_scalar = 3.0

    prev_score_reward = 0
    prev_remaining_pellets = 0
    prev_level = 0
    prev_lives = 3


    # Pac-Man specific buttons (only directional controls)
    gameboy_buttons = [
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_ARROW_DOWN
    ]
    gameboy_buttons_release = [
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_ARROW_DOWN
    ]

    def __init__(self, rom_path, observation_shape=None, window_type="headless"):
        assert os.path.exists(rom_path), f"ROM {rom_path} not found"
        self.pyboy = PyBoy(rom_path)
        self.initialize_game()
        self.action_space = spaces.Discrete(len(self.gameboy_buttons))
        self.observation_shape = observation_shape if observation_shape else self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(*self.observation_shape, 3), dtype=np.uint8)
        print(f"Observation shape: {self.observation_shape}")

    def initialize_game(self):
        # Similar to Mario, just pressing START to begin the game
        self.pyboy.set_emulation_speed(0)
        for _ in range(500):
            self.pyboy.tick()

        state_path = "roms/initial_state_pacman.state"
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for _ in range(100):
            self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        if not os.path.exists(state_path):
            self.pyboy.save_state(open(state_path, "wb"))

    def step(self, action):
        # Similar structure to the original, with adaptations for Pac-Man
        if action < len(self.gameboy_buttons):
            self.pyboy.send_input(self.gameboy_buttons[action])

        for _ in range(self.n_ticks_per_step):#plus one to account for main tick
            self.pyboy.tick()

        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
        observation = cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
        self.pyboy.send_input(self.gameboy_buttons_release[action])
        self.pyboy.tick()
        self.time_steps += 1
        done = self.check_game_over()
        reward, info = self.calculate_reward(observation, done)
        

        return observation, reward, done, info

    def calculate_reward(self, observation, done):
        reward = 0

        # Extract BCD-encoded score from memory
        score_bcd_bytes = [self.pyboy.get_memory_value(0x0070 + i) for i in range(6)]
        current_score = bcd_to_int(score_bcd_bytes)
        
        # Calculate score delta
        score_delta = current_score - self.prev_score_reward
        reward += score_delta * self.score_scalar

        # Calculate pellets delta
        remaining_pellets = self.pyboy.get_memory_value(0x006A)
        pellets_delta = self.prev_remaining_pellets - remaining_pellets
        reward += pellets_delta * self.pellet_scalar

        # Calculate level progression
        current_level = self.pyboy.get_memory_value(0x0068)
        if current_level > self.prev_level:
            reward += (current_level - self.prev_level) * self.level_scalar

        # Check for lives lost
        current_lives = self.pyboy.get_memory_value(0x0067)
        if current_lives < self.prev_lives:
            reward += self.death_scalar

        # Update previous state for the next reward calculation
        self.prev_score_reward = current_score
        self.prev_remaining_pellets = remaining_pellets
        self.prev_level = current_level
        self.prev_lives = current_lives

        print("current_score",current_score,"current_level",current_level,"current_lives",current_lives,"remaining_pellets",remaining_pellets,"reward",reward)
        return reward, {
            "score_delta": score_delta,
            "pellets_delta": pellets_delta,
            "level_progression": (current_level - self.prev_level) * self.level_scalar if current_level > self.prev_level else 0,
            "lives_lost": self.death_scalar if current_lives < self.prev_lives else 0
        }

    def check_game_over(self):
        # Adapted to check for game over conditions in Pac-Man
        lives = self.pyboy.get_memory_value(0x0067)
        return lives < 1 or self.time_steps >= self.max_steps

    def reset(self):
        # Loading initial state for a fresh game start
        self.pyboy.load_state(open("roms/initial_state_pacman.state", "rb"))
        self.time_steps = 0
        self.prev_score_reward = 0
        self.prev_remaining_pellets = self.pyboy.get_memory_value(0x006A)
        self.prev_level = self.pyboy.get_memory_value(0x0068)
        self.prev_lives = self.pyboy.get_memory_value(0x0067)
        return cv2.resize(self.pyboy.botsupport_manager().screen().screen_ndarray(), self.observation_shape, interpolation=cv2.INTER_AREA)

    def render(self, mode='human'):
        # Displaying the game screen
        img = self.pyboy.botsupport_manager().screen().screen_ndarray()
        if mode == 'human':
            cv2.imshow("Pac-Man", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return img

    def close(self):
        self.pyboy.stop()
        
        
def bcd_to_int(bcd_bytes):
    """
    Convert a sequence of BCD (Binary-Coded Decimal) bytes to an integer.
    bcd_bytes: A list or tuple of bytes representing the BCD value, 
               ordered from most significant byte to least significant byte.
    """
    total_value = 0
    # Process each BCD byte starting from the most significant byte
    for byte in bcd_bytes:
        # Shift the total value by a decimal place (multiply by 10) for each nibble (4 bits) in the byte
        for nibble_shift in (4, 0):
            nibble_value = (byte >> nibble_shift) & 0xF
            total_value = total_value * 10 + nibble_value
    return total_value