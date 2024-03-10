# %%

import torch

# %%
from agents.DQNagent import *
from agents.PPOagent import *
from gameboy_env import *
import sys
sys.path.append('models')
from models.dqn import *

# %%
print(torch.cuda.is_available())
env = MarioGameBoyEnv("roms/Super Mario Land (JUE) (V1.1) [!].gb",observation_shape=(84,84), window_type="human")
# Configuration with the game name and hyperparameters
game_name = "SuperMarioLand"
config = Config(
    game_name=game_name,
    env=env,
    model_function=DQN2,
    target_model_function=DQN2,
    replay_buffer_size=100000,  # Optional, showing default value as an example
    gamma=0.9999999,  # Optional, showing default value as an example
    batch_size=32,
    epsilon_decay=.999,  # Optional, showing default value as an example
    epsilon_decay_frames=1000000,  # Optional, showing default value as an example
    epsilon_min=0.1,  # Optional, showing default value as an example
    epsilon=1,
    state_preprocessing_fn=None,  # Assuming a default value; define if needed
    update_target_every=50,
    save_model_every=50,
    n_ticks_to_skip=4,
    epsilon_decay_rate_adjustment=1  # Optional, showing default value as an example
)
# Initialize the DQN Agent
agent = DQNAgent(config) 

# %%
agent.train(100000)


# %%
state, action, reward, next_state, done=agent.replay_buffer.sample(1)

# %%
plt.imshow(state.squeeze(0).transpose(1, 2, 0))

# %%
plt.imshow(next_state.squeeze(0).transpose(1, 2, 0))

# %%
print(action[0], reward, done)

# %%



