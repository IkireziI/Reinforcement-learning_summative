import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import datetime

# Ensure the models directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Policy Network Architecture
class PolicyNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super(PolicyNetwork, self).__init__()
        # Assuming observation is a 1D vector like in DeliveryRobotEnv
        input_dim = observation_space_shape[-1] 

        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        # Output layer produces logits for a categorical distribution over actions
        self.fc3 = nn.Linear(128, action_space_n)

    def forward(self, state):
        # Flatten the state if it's 3D (e.g., [batch_size, 1, 7] -> [batch_size, 7])
        # This handles cases where Gym's reset might return state with an extra dimension
        if state.dim() == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x) # Return logits

# REINFORCE Training Function
def train_pg(env_name="DeliveryRobot-v0", num_episodes=1000,
             render_mode=None, model_save_path="pg_robot_model"):
    
    env = gym.make(env_name, render_mode=render_mode)
    observation_space_shape = env.observation_space.shape
    action_space_n = env.action_space.n

    # Hyperparameters for REINFORCE
    LR = 0.001
    GAMMA = 0.99 # Discount factor

    # Initialize Policy Network and Optimizer
    policy_net = PolicyNetwork(observation_space_shape, action_space_n)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Training logs
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    best_avg_reward = -float('inf') # To track the best performing model
    best_model_path = None

    print("Starting Policy Gradient (REINFORCE) training for {} episodes...".format(num_episodes))

    for i_episode in range(1, num_episodes + 1):
        log_probs = []
        rewards_history = []
        
        state, info = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension
        done = False
        truncated = False
        total_reward = 0
        steps_count = 0

        while not done and not truncated:
            # Get action probabilities from policy network
            logits = policy_net(state)
            action_distribution = Categorical(logits=logits)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)

            next_state, reward, done, truncated, info = env.step(action.item())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            total_reward += reward
            steps_count += 1

            log_probs.append(log_prob)
            rewards_history.append(reward)

            state = next_state

            if done or truncated:
                break
        
        # Calculate discounted rewards (returns)
        returns = []
        G = 0
        for r in reversed(rewards_history):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalize returns (optional, but often helps stability)
        # if len(returns) > 1: # Avoid division by zero if episode length is 1
        #     returns = (returns - returns.mean()) / (returns.std() + 1e-9)


        # Calculate policy loss
        policy_loss = []
        for lp, G in zip(log_probs, returns):
            policy_loss.append(-lp * G) # Negative because we want to maximize rewards

        optimizer.zero_grad()
        # Sum all loss components and backpropagate
        loss = torch.cat(policy_loss).sum() 
        loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps_count)

        # Logging and model saving
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])

            # Success rate calculation (assuming success > 900 reward)
            current_successes = 0
            for k in range(1, 11):
                if i_episode - k >= 0 and episode_rewards[i_episode - k] >= 900:
                    current_successes += 1
            success_rate = (current_successes / 10) * 100
            success_rates.append(success_rate)


            print(f"Episode {i_episode}/{num_episodes} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate (last 10): {success_rate:.1f}%")

            # Save the BEST model based on average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                best_model_filename = f"{model_save_path}_best_{timestamp}.pth"
                best_model_path = os.path.join(MODELS_DIR, best_model_filename)
                torch.save(policy_net.state_dict(), best_model_path)
                print(f"--> NEW BEST PG Model Saved: {best_model_path} (Avg Reward: {best_avg_reward:.2f})")

    env.close()
    print("Policy Gradient training finished.")

    # Plotting results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode (Policy Gradient)")

    plt.subplot(1, 2, 2)
    rolling_avg_rewards = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
    plt.plot(rolling_avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Rolling Average Reward (Window 50) (Policy Gradient)")

    plt.tight_layout()
    plt.show()

    # Save the FINAL model
    if model_save_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_model_path = os.path.join(MODELS_DIR, f"{model_save_path}_final_{timestamp}.pth")
        torch.save(policy_net.state_dict(), final_model_path)
        print(f"FINAL PG Model Saved to {final_model_path}")

    return best_model_path, final_model_path # Return paths for potential zipping later