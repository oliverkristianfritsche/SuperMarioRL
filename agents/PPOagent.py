import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Categorical
from models import *
from utils import ReplayBuffer
from matplotlib import pyplot as plt
from IPython.display import clear_output
from torch import optim

class PPO2Agent:
    def __init__(self,env,model_function, target_model_function, checkpoint_subfolder="./checkpoints/PPO2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_function(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = target_model_function(env.observation_space.shape, env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.env = env
        self.checkpoint_dir = checkpoint_subfolder
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(10000)
        # PPO specific attributes
        self.clip_param = 0.2
        self.ppo_update_times = 10
        self.batch_size = 64
        # Lists to store rewards, epsilons, time steps, and gammas for plotting
        self.total_rewards = []
        self.epsilons = []
        self.total_time_steps = []
        self.total_gammas = []


    def act(self, state, visualize=False):
        # Optional visualization of the state
        if visualize:
            plt.imshow(state)
            plt.show()

        # Assuming state is a NumPy array with shape (H, W, C)
        # Transpose state to match PyTorch's expected (C, H, W) format
        state = np.transpose(state, (2, 0, 1))

        # Normalize the state if your model expects inputs in a certain range, e.g., [0, 1] or [-1, 1]
        # Example for normalization to [0, 1]
        state = state / 255.0
      
        # Convert state to PyTorch tensor and add batch dimension [B, C, H, W]
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Forward pass through the model to get action probabilities
        with torch.no_grad():
            action_probs,_ = self.model(state_t)

        # Assuming action_probs are logits, use softmax to get probabilities
        action_probs = F.softmax(action_probs, dim=-1)

        # Sample an action from the probabilities
        action = Categorical(action_probs).sample().item()

        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            # Not enough samples to perform the update
            return

        for _ in range(self.ppo_update_times):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            states = states.transpose(0, 3, 1, 2)
            next_states = next_states.transpose((0, 3, 1, 2))
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            print(f"states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, next_states: {next_states.shape}, dones: {dones.shape}")
            # Get model outputs for next and current states
            _,next_state_values = self.model(next_states)
            old_action_probs,current_state_values = self.model(states)

            # Assuming the unexpected dimension is the last one, and we want to average it
            next_values = next_state_values.mean(dim=-1).squeeze()
            current_values = current_state_values.mean(dim=-1).squeeze()

            # print(f"rewards: {rewards.shape}, next_values: {next_values.shape}, dones: {dones.shape}, current_values: {current_values.shape}")

            # Compute TD target and advantages
            td_target = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_target - current_values

            # Calculate old log probs
            old_log_probs =torch.log(old_action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)

            # Optimize policy:
            action_probs, state_values = self.model(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs.detach())

            # Compute PPO objective and value loss
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            if state_values.dim() > 1:
                state_values = state_values.squeeze(1)
            value_loss = F.mse_loss(td_target.detach(), state_values)

            # Take gradient step
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()

    def plot(self):
        clear_output(wait=True)
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.title("Total Reward per Episode")
        plt.plot(self.total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.subplot(2, 2, 2)
        plt.title("Epsilon Value Over Time")
        plt.plot(self.epsilons)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.subplot(2, 2, 3)
        plt.title("Time Steps per Episode")
        plt.plot(self.total_time_steps)
        plt.xlabel("Episode")
        plt.ylabel("Time Steps")

        plt.subplot(2, 2, 4)
        plt.title("Gamma per Episode")
        plt.plot(self.total_gammas)
        plt.xlabel("Episode")
        plt.ylabel("Gamma")

        plt.tight_layout()
        plt.show()

    def train(self, num_episodes, start=0, reward_discount_factor=0.99, discount_step_interval=100):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_path = os.path.join(self.checkpoint_dir, f"model_{start}.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.target_model.load_state_dict(torch.load(model_path))

        for episode in range(start, num_episodes):
            state = self.env.reset()
            print(state.shape)
            total_reward = 0
            done = False
            self.time_steps = 0
            self.temp_gamma = self.gamma
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                if self.time_steps % discount_step_interval == 0:
                    self.temp_gamma *= reward_discount_factor

                if (reward > 100):
                    continue
                state = next_state
                total_reward += reward * self.temp_gamma

                self.update()
                self.time_steps += 1

            self.total_rewards.append(total_reward)
            self.epsilons.append(self.epsilon)
            self.total_time_steps.append(self.time_steps)
            self.total_gammas.append(self.temp_gamma)

            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            if episode % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_{episode}.pth"))
                torch.save(self.target_model.state_dict(), os.path.join(self.checkpoint_dir, f"target_model_{episode}.pth"))

            print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {self.epsilon}, Time Steps: {self.time_steps}, Gamma: {self.temp_gamma}")
