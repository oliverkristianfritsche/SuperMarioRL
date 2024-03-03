import gymnasium as gym
from pyboy import PyBoy, WindowEvent
from gym import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import FrameStorage, bcd_to_int
import cv2
class GameBoyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    gameboy_buttons = [
        WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT,
    ]
    # WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A,
    #     WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT,
    #     WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B,
    #     WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT,

    time_steps = 0
    death_scalar=0
    survive_scalar=100
    frame_scalar=0.000
    level_scalar=0.0
    coin_scalar=1.00
    score_scalar=.01
    max_steps=4000

    prev_level_reward = 0
    prev_coin_reward = 0
    prev_score_reward = 0


    def __init__(self, rom_path, observation_shape=None, window_type="headless"):
        super(GameBoyEnv, self).__init__()
        assert os.path.exists(rom_path), f"ROM {rom_path} not found"
        self.pyboy = PyBoy(rom_path)
        self.initialize_game()
        self.action_space = spaces.Discrete(len(self.gameboy_buttons))
        self.observation_shape = observation_shape if observation_shape else self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(*self.observation_shape, 3), dtype=np.uint8)
        self.unique_frames = FrameStorage()

        print(f"Observation shape: {self.observation_shape}")

    def initialize_game(self):
        self.pyboy.set_emulation_speed(0)
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        for _ in range(100):
            self.pyboy.tick()

        state_path = "roms/initial_state.state"
        if not os.path.exists(state_path):
            self.pyboy.save_state(open(state_path, "wb"))

    def step(self, action):
        if action < len(self.gameboy_buttons):
            self.pyboy.send_input(self.gameboy_buttons[action])
        self.pyboy.tick()

        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
    
        # Resize observation to the specified shape
        observation = cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
        
        done = self.check_game_over()
        reward, info = self.calculate_reward(observation, done)

        return observation, reward, done, info

    def calculate_reward(self, observation, done):
        # Get current rewards
        current_level_reward = self.get_level_reward()
        current_coin_reward = self.get_coin_reward()
        current_score_reward = self.get_score_reward()

        # Calculate deltas
        delta_level_reward = current_level_reward - self.prev_level_reward
        delta_coin_reward = current_coin_reward - self.prev_coin_reward
        delta_score_reward = current_score_reward - self.prev_score_reward

        if(delta_coin_reward<0):
            delta_coin_reward=0
        if(delta_level_reward<0):
            delta_level_reward=0
        if(delta_score_reward<0):
            delta_score_reward=0

        # Update the frame reward
        frame_reward = self.unique_frames.add_frame(observation) * self.frame_scalar

        # Total reward is the sum of frame reward and deltas
        reward = frame_reward + delta_level_reward + delta_coin_reward + delta_score_reward

        # Update previous reward values for the next step
        self.prev_level_reward = current_level_reward
        self.prev_coin_reward = current_coin_reward
        self.prev_score_reward = current_score_reward

        if done:
            print(f"Level Delta: {delta_level_reward}, Coin Delta: {delta_coin_reward}, Score Delta: {delta_score_reward}, Total Reward: {reward}")

        return reward, {"level_reward": delta_level_reward, "coin_reward": delta_coin_reward, "score_reward": delta_score_reward}
    
    def check_death_animation(self):
        return self.pyboy.get_memory_value(0xFFA6) == 0x90

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
    
    def get_score(self):
        score_bytes = [self.pyboy.get_memory_value(addr) for addr in (0xC0A0, 0xC0A1, 0xC0A2)]
        score = bcd_to_int(score_bytes)
        return score

    def get_score_reward(self):
        return (self.get_score()-(100*self.pyboy.get_memory_value(0xFFFA))) * self.score_scalar  # Adjust the reward scaling as needed (self.get_score()-(100*self.pyboy.get_memory_value(0xFFFA))) * self.score_scalar 

    def reset(self):
        self.pyboy.load_state(open("roms/initial_state.state", "rb"))
        self.unique_frames = FrameStorage()
        observation = self.pyboy.botsupport_manager().screen().screen_ndarray()
        # Resize observation to the specified shape
        observation = cv2.resize(observation, self.observation_shape, interpolation=cv2.INTER_AREA)
        return observation
    def check_game_over(self):
        
        return self.pyboy.get_memory_value(0xDA15) <2 or self.time_steps > self.max_steps

    def close(self):
        self.pyboy.stop()

    def render(self, mode='human'):
        img = self.pyboy.botsupport_manager().screen().screen_ndarray()
        if mode == 'human':
            plt.imshow(img)
            plt.show()
        return img if mode == 'rgb_array' else None
        
        
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