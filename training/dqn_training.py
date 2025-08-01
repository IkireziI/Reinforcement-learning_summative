import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import datetime

# Ensure the models directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super(QNetwork, self).__init__()
        input_dim = observation_space_shape[-1] 

        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_n)

    def forward(self, state):
        if state.dim() == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.stack(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.stack(next_states),
            torch.BoolTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

# DQN Training Function
def train_dqn(env_name="DeliveryRobot-v0", num_episodes=1000,
              render_mode=None, model_save_path=None):

    env = gym.make(env_name, render_mode=render_mode)
    observation_space_shape = env.observation_space.shape
    action_space_n = env.action_space.n

    # Hyperparameters
    LR = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.99
    
    REPLAY_BUFFER_CAPACITY = 5000 
    BATCH_SIZE = 32
    
    TARGET_UPDATE_FREQ = 10

    # Initialize Q-Networks
    policy_net = QNetwork(observation_space_shape, action_space_n)
    target_net = QNetwork(observation_space_shape, action_space_n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Replay Buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # Training logs
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    epsilon = EPSILON_START

    # --- NEW: Best model tracking for DQN ---
    best_avg_reward = -float('inf')
    best_dqn_model_path = None
    # --- END NEW ---

    print("Starting DQN training for {} episodes...".format(num_episodes))

    for i_episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        truncated = False
        total_reward = 0
        steps_count = 0
        episode_success = False

        while not done and not truncated:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            total_reward += reward
            steps_count += 1

            if done and reward > 0:
                episode_success = True

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            if len(replay_buffer) > BATCH_SIZE:
                experiences = replay_buffer.sample(BATCH_SIZE)
                if experiences is not None:
                    states, actions, rewards, next_states, dones = experiences

                    q_values_output = policy_net(states)
                    actions_unsqueezed = actions.unsqueeze(1)
                    
                    current_q_values = q_values_output.gather(1, actions_unsqueezed).squeeze(1)

                    next_q_values = target_net(next_states).max(1)[0].detach()
                    expected_q_values = rewards + (GAMMA * next_q_values * (~dones))

                    loss = nn.MSELoss()(current_q_values, expected_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if done or truncated:
                break

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps_count)

        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])

            current_successes = 0
            for k in range(1, 11):
                if i_episode - k >= 0 and episode_rewards[i_episode - k] >= 900:
                    current_successes += 1
            success_rate = (current_successes / 10) * 100
            success_rates.append(success_rate)


            print(f"Episode {i_episode}/{num_episodes} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Success Rate (last 10): {success_rate:.1f}%")

            if i_episode % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
            
            # --- NEW: Save the BEST DQN model ---
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                best_model_filename = f"{model_save_path}_best_{timestamp}.pth"
                best_dqn_model_path = os.path.join(MODELS_DIR, best_model_filename)
                torch.save(policy_net.state_dict(), best_dqn_model_path)
                print(f"--> NEW BEST DQN Model Saved: {best_dqn_model_path} (Avg Reward: {best_avg_reward:.2f})")
            # --- END NEW ---

    env.close()
    print("Training finished.")

    # Plotting results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode (DQN)") # Added (DQN) for clarity

    plt.subplot(1, 2, 2)
    rolling_avg_rewards = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
    plt.plot(rolling_avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Rolling Average Reward (Window 50) (DQN)") # Added (DQN) for clarity

    plt.tight_layout()
    plt.show()

    # --- NEW: Save the FINAL DQN model and return paths ---
    final_dqn_model_path = None
    if model_save_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_model_filename = f"{model_save_path}_final_{timestamp}.pth"
        final_dqn_model_path = os.path.join(MODELS_DIR, final_model_filename)
        torch.save(policy_net.state_dict(), final_dqn_model_path)
        print(f"FINAL DQN Model Saved to {final_dqn_model_path}")
    
    return best_dqn_model_path, final_dqn_model_path # Return both paths
    # --- END NEW ---