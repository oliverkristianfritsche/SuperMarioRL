import matplotlib.pyplot as plt
from IPython.display import clear_output
from models import *
from utils import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from pyboy import WindowEvent
class Config:
    def __init__(self, env, model_function, target_model_function, **kwargs):
        self.env = env
        self.model_function = model_function
        self.target_model_function = target_model_function
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 100000)
        self.gamma = kwargs.get('gamma', 0.9999999)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epsilon_decay = kwargs.get('epsilon_decay', .999)
        self.epsilon_decay_frames = kwargs.get('epsilon_decay_frames', 1000000)
        self.epsilon_min = kwargs.get('epsilon_min', 0.1)
        self.epsilon = kwargs.get('epsilon', 1)
        self.state_preprocessing_fn = kwargs.get('state_preprocessing_fn', None)
        self.update_target_every = kwargs.get('update_target_every', 30)
        self.save_model_every = kwargs.get('save_model_every', 100)
        self.n_ticks_to_skip = kwargs.get('n_ticks_to_skip', 2)
        self.state_preprocessing_fn = kwargs.get('state_preprocessing_fn', None)
        self.epsilon_decay_rate_adjustment = kwargs.get('epsilon_decay_rate_adjustment', 1)
        self.checkpoint_subfolder = kwargs.get('checkpoint_subfolder', '')

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = os.path.join("checkpoints/DQN", config.checkpoint_subfolder)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_metrics()
        self.load_training_history()
        
        # Preprocessing function
        self.preprocess_state = config.state_preprocessing_fn if config.state_preprocessing_fn else self.default_preprocess_state


        # Model and target model initialization
        self.model = config.model_function(config.env.observation_space.shape, config.env.action_space.n).to(self.device)
        self.target_model = config.target_model_function(config.env.observation_space.shape, config.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

        # Load checkpoint if available
        self.load_checkpoint(prefix='model')  # Loads the latest 'model'
        self.load_checkpoint(prefix='target_model')  # Loads the latest 'target_model'

        

    def initialize_metrics(self):
        self.total_rewards = []
        self.epsilons = []
        self.total_time_steps = []
        self.total_gammas = []
        self.time_steps = 0
        self.minibatch_update_count = 0
        self.epoch_counter = 0
        self.episode_counter = 0

    def load_checkpoint(self, prefix=None):
        """
        Load the most recently modified model file that matches the given prefix.
        
        :param prefix: Optional. The prefix of the model files to consider. If None, considers all files.
        """
        # Filter files by prefix, if provided
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if prefix:
            checkpoint_files = [f for f in checkpoint_files if f.startswith(prefix)]
        
        if not checkpoint_files:
            print(f"No checkpoint found for prefix '{prefix}', starting from scratch.")
            return

        # Sort files by modification time in descending order
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        latest_model_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])

        # Load the most recently modified model that matches the prefix
        model_to_load = torch.load(latest_model_checkpoint, map_location=self.device)
        if prefix and 'model' in prefix:
            self.model.load_state_dict(model_to_load)
        elif prefix and 'target_model' in prefix:
            self.target_model.load_state_dict(model_to_load)
        
        print(f"Loaded checkpoint from {latest_model_checkpoint} for prefix '{prefix}'")

    def plot_metrics_and_filters(self):
        clear_output(wait=True)
        fig = plt.figure(figsize=(20, 20))  # Increased figure size to accommodate additional plots

        # Determine the number of complete epochs for even distribution
        episodes_per_epoch = len(self.total_rewards) // (self.epoch_counter + 1)
        total_episodes_to_consider = episodes_per_epoch * (self.epoch_counter + 1)

        # Calculate the average for rewards and gammas per epoch
        average_rewards_per_epoch = [sum(self.total_rewards[i:i + episodes_per_epoch]) / episodes_per_epoch for i in range(0, total_episodes_to_consider, episodes_per_epoch)]
        average_gammas_per_epoch = [sum(self.total_gammas[i:i + episodes_per_epoch]) / episodes_per_epoch for i in range(0, total_episodes_to_consider, episodes_per_epoch)]

        # Adjust titles and metrics for plotting, including per-episode data
        titles = ["Reward per Episode", "Average Reward per Epoch", "Gamma per Episode", "Average Gamma per Epoch",
                "Epsilon Value Over Time", "Time Steps per Episode"]
        metrics = [self.total_rewards, average_rewards_per_epoch, self.total_gammas, average_gammas_per_epoch,
                self.epsilons, self.total_time_steps]

        for i, (title, metric) in enumerate(zip(titles, metrics)):
            ax = fig.add_subplot(4, 2, i+1)  # Adjusted for 4x2 grid layout
            ax.set_title(title)
            ax.plot(metric)

        # Assuming the first layer of your model is a convolutional layer
        first_conv_layer = next(self.model.children())
        if isinstance(first_conv_layer, torch.nn.modules.conv.Conv2d):
            filters = first_conv_layer.weight.data.clone().cpu()
            n_filters = filters.shape[0]
            n_display_filters = min(n_filters, 4)  # Display the first 4 filters

            filters_ax = fig.add_subplot(4, 2, 7)  # Adjusted for the new layout
            filters_ax.set_title("First Layer Filters")
            filters_ax.axis('off')

            for i in range(n_display_filters):
                inner_ax = fig.add_subplot(4, n_display_filters, i + 1 + 3*n_display_filters)
                filter = filters[i]
                filter_min, filter_max = filter.min(), filter.max()
                filter = (filter - filter_min) / (filter_max - filter_min)
                inner_ax.imshow(filter.permute(1, 2, 0))
                inner_ax.axis('off')

        plt.tight_layout()
        plt.show()


    def default_preprocess_state(self,state):
        # Convert state to PyTorch tensor and permute dimensions to match [C, H, W] format expected by PyTorch Conv2D layers
        return torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)

    def act(self, state, visualize=False):
        if random.random() < self.config.epsilon:
            return self.config.env.action_space.sample()
        state_t = self.preprocess_state(state).to(self.device)
        if visualize:
            plt.imshow(state_t.cpu().squeeze().permute(1, 2, 0).numpy())
            plt.show()
        q_values = self.model(state_t)
        return q_values.max(1)[1].item()

    def plot_metrics(self):
        clear_output(wait=True)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        titles = ["Total Reward per Episode", "Epsilon Value Over Time", "Time Steps per Episode", "Gamma per Episode"]
        metrics = [self.total_rewards, self.epsilons, self.total_time_steps, self.total_gammas]

        for ax, title, metric in zip(axs.flat, titles, metrics):
            ax.set_title(title)
            ax.plot(metric)

        plt.tight_layout()
        plt.show()

    def save_model(self, prefix="model", episode=None):
        model_path = os.path.join(self.checkpoint_dir, f"{prefix}_{episode if episode is not None else self.epoch_counter}.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, f"training_history_{episode if episode is not None else self.epoch_counter}.pth")
        history = {
            'total_rewards': self.total_rewards,
            'epsilons': self.epsilons,
            'total_time_steps': self.total_time_steps,
            'total_gammas': self.total_gammas,
            'last_episode': episode,
            'last_epoch': self.epoch_counter
        }
        torch.save(history, history_path)

    def load_training_history(self):
        # List all training history files
        history_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('training_history_')]
        
        # Sort files by last modification time in descending order
        history_files_sorted = sorted(history_files, key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        
        if history_files_sorted:
            latest_history_checkpoint = os.path.join(self.checkpoint_dir, history_files_sorted[0])
            history = torch.load(latest_history_checkpoint, map_location=self.device)
            
            self.total_rewards = history['total_rewards']
            self.epsilons = history['epsilons']
            self.total_time_steps = history['total_time_steps']
            self.total_gammas = history['total_gammas']
            self.episode_counter = history['last_episode']
            self.epoch_counter = history['last_epoch'] if 'last_epoch' in history else 6
            
            print(f"Loaded training history from {latest_history_checkpoint}")
        else:
            print("No training history found, starting from scratch.")

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(-1).to(self.device)  # Ensure correct shape [batch_size, 1]
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        #print("State shape:",state.shape,"Action shape:",action.shape,"Reward shape:",reward.shape,"Next state shape:",next_state.shape,"Done shape:",done.shape)

        
        q_values = self.model(state)
        #print("q_values shape:", q_values.shape)  # Expected to be [64, num_actions]

        next_q_values = self.target_model(next_state)
        #print("next_q_values shape:", next_q_values.shape)  # Expected to be [64, num_actions]

        reward = reward.unsqueeze(1)  # This changes shape from [64] to [64, 1]
        done = done.unsqueeze(1)  # This changes shape from [64] to [64, 1]

        # Now, when you perform the operation, the shapes should be compatible
        expected_q_values = reward + next_q_values * (1 - done)
        #print("Expected Q values shape:",expected_q_values.shape)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.minibatch_update_count += 1
        if self.minibatch_update_count % 50000 == 0:
            self.epoch_counter += 1
            self.target_model.load_state_dict(self.model.state_dict())
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_epoch_{self.epoch_counter}.pth"))
            torch.save(self.target_model.state_dict(), os.path.join(self.checkpoint_dir, f"target_model_epoch_{self.epoch_counter}.pth"))
            #print(f"Epoch: {self.epoch_counter} completed.")

    def train(self, num_episodes, reward_discount_factor=0.95, discount_step_interval=200):
        start_episode = self.episode_counter + 1  # Assuming episode_counter is correctly initialized
        epsilon_decay_amount = (self.config.epsilon - self.config.epsilon_min) / self.config.epsilon_decay_frames
        self.config.epsilon = self.epsilons[-1] if self.epsilons else self.config.epsilon
        total_frames = 0

        for episode in range(start_episode, start_episode + num_episodes):
            state = self.config.env.reset()
            total_reward = 0
            done = False
            self.time_steps = 0
            self.temp_gamma = self.config.gamma  # Reset temp_gamma at the start of each episode

            if self.config.env.action_space.n == 2: #special case for 1dof env
                self.config.env.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                self.config.env.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            if self.config.env.action_space.n == 4: #special case for 2dof env
                self.config.env.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.config.env.step(action)
                self.time_steps += 1
                total_frames += 1
                if reward > 100:
                    continue
                # Adjusting reward with current temp_gamma before storing in replay buffer
                adjusted_reward = (reward * self.temp_gamma) + (-self.config.env.death_scalar * (1 - self.temp_gamma))
                self.replay_buffer.push(state.transpose(2, 0, 1), action, adjusted_reward, next_state.transpose(2, 0, 1), done)
                self.update()
                state = next_state

                
                total_reward += reward

                # Apply reward discount factor adjustment every 'discount_step_interval' steps
                if self.time_steps % discount_step_interval == 0:
                    self.temp_gamma *= reward_discount_factor

                # Epsilon decay logic should be based on total_frames, not self.time_steps
                if total_frames <= self.config.epsilon_decay_frames:
                    self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon - epsilon_decay_amount)
                else:
                    self.config.epsilon = self.config.epsilon_min

            # Log and save metrics after each episode
            self.total_rewards.append(total_reward)
            self.epsilons.append(self.config.epsilon)
            self.total_time_steps.append(self.time_steps)
            self.total_gammas.append(self.temp_gamma)

            if episode % self.config.update_target_every == 0:  # Fixed condition to match episode interval
                self.target_model.load_state_dict(self.model.state_dict())
            
            if episode % self.config.save_model_every == 0:  # Fixed condition to match episode interval
                self.save_model(prefix="model",episode=episode)
                self.save_model(prefix="target_model",episode=episode)
            self.plot_metrics_and_filters()
            print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {self.config.epsilon}, Time Steps: {self.time_steps}, Gamma: {self.temp_gamma}")


                
