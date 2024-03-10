from collections import deque
import numpy as np
import random
import torch
import pickle
import glob
import os
from math import ceil
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def save(self, folderpath, chunk_size=1000):
        """Save the replay buffer to multiple files in chunks."""
        total_items = len(self.buffer)
        num_chunks = ceil(total_items / chunk_size)
        
        for i in range(num_chunks):
            chunk = list(self.buffer[i*chunk_size:(i+1)*chunk_size])
            filename = os.path.joing(folderpath,f"chunk_{i}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(chunk, file)

    def load(self, folderpath):
        """Load the replay buffer from multiple chunk files."""
        chunk_files = glob.glob(os.path.join(folderpath, 'chunk_*.pkl'))
        buffer = []
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as file:
                chunk = pickle.load(file)
                buffer.extend(chunk)
        
        self.buffer = deque(buffer, maxlen=self.buffer.maxlen)

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
