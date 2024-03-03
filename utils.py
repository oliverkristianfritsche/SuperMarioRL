from collections import deque
import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class FrameStorage:
        def __init__(self):
            self.unique_frame_hashes = set()

        def __len__(self):
            # Return the number of unique frames
            return len(self.unique_frame_hashes)

        def add_frame(self, observation):
            #crop the top 35% of the screen
            observation = observation[35:,:,:]

            # Convert the frame to a byte string and then hash it
            frame_hash = hash(observation.tostring())
            reward = 1 if frame_hash not in self.unique_frame_hashes else 0
            self.unique_frame_hashes.add(frame_hash)
            return reward
        

def bcd_to_int(bcd_bytes):
    total_value = 0
    for byte in bcd_bytes:
        for nibble_shift in (4, 0):
            nibble_value = (byte >> nibble_shift) & 0xF
            total_value = total_value * 10 + nibble_value
    return total_value

def default_preprocess_state(state):
    # Convert state to PyTorch tensor and permute dimensions to match [C, H, W] format expected by PyTorch Conv2D layers
    return torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
