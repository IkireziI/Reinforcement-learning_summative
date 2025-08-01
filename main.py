import gymnasium as gym
from gymnasium.envs.registration import register
import os
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy as np
import zipfile

# Import your custom environment class
from environment.custom_env import DeliveryRobotEnv 

# Import training functions and network classes
from training.dqn_training import train_dqn, QNetwork
# from training.pg_training import train_pg, PolicyNetwork # COMMENTED OUT FOR NOW

# --- Environment Registration ---
try:
    gym.make("DeliveryRobot-v0")
except Exception:
    register(
        id="DeliveryRobot-v0",
        entry_point="environment.custom_env:DeliveryRobotEnv",
        kwargs={"size": 10},
        max_episode_steps=500,
        reward_threshold=900,
    )
# --- End Environment Registration ---

# --- Utility Function for Evaluation ---
def evaluate_model(model_path, network_class, env_name="DeliveryRobot-v0", num_eval_episodes=5, render=True):
    """
    Evaluates a trained model in the environment.
    network_class: Either QNetwork or PolicyNetwork
    """
    print(f"\n--- Starting Evaluation of Model: {model_path} ---")
    
    env = gym.make(env_name, render_mode='human' if render else None)
    
    observation_space_shape = env.observation_space.shape
    action_space_n = env.action_space.n

    policy_net = network_class(observation_space_shape, action_space_n)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval() # Set to evaluation mode

    eval_rewards = []

    for i_episode in range(num_eval_episodes):
        state, info = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            with torch.no_grad():
                if issubclass(network_class, QNetwork): # If it's a DQN
                    q_values = policy_net(state)
                    action = q_values.argmax().item()
                # elif issubclass(network_class, PolicyNetwork): # PG branch is commented out for now
                #     logits = policy_net(state)
                #     action_distribution = torch.distributions.Categorical(logits=logits)
                #     action = action_distribution.sample().item()
                else:
                    # This should not be reached if PG is commented out and only QNetwork is used
                    raise ValueError("Unknown network class for evaluation or PG branch not active.")

            next_state, reward, done, truncated, info = env.step(action)
            state = torch.FloatTensor(next_state).unsqueeze(0)
            total_reward += reward

            if done or truncated:
                break
        
        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {i_episode+1}/{num_eval_episodes}: Total Reward = {total_reward:.2f}")

    env.close()
    print(f"--- Evaluation Complete. Average Reward over {num_eval_episodes} episodes: {np.mean(eval_rewards):.2f} ---")
    return np.mean(eval_rewards) # Return average reward for potential use

# --- Utility Function for Zipping Models ---
def zip_models(algo_name, model_type, model_paths):
    """
    Zips specific model files.
    algo_name: 'dqn' or 'pg'
    model_type: 'best' or 'final'
    model_paths: list of full paths to the .pth files to zip
    """
    zip_filename = f"{algo_name}_{model_type}_model.zip"
    zip_filepath = os.path.join("models", zip_filename)
    
    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for model_path in model_paths:
                if os.path.exists(model_path):
                    zipf.write(model_path, os.path.basename(model_path)) # Add to zip with just filename
                    print(f"Added {os.path.basename(model_path)} to {zip_filename}")
                else:
                    print(f"Warning: Model file not found and skipped: {model_path}")
        print(f"Successfully created {zip_filepath}")
        return zip_filepath
    except Exception as e:
        print(f"Error zipping models for {algo_name} {model_type}: {e}")
        return None


# --- Main Experiment Running Function (DQN ONLY) ---
def run_dqn_experiment(): # Renamed for clarity
    env_name = "DeliveryRobot-v0"
    num_episodes = 1000
    render_mode = None 
    
    # --- DQN Experiment ---
    print("\n" + "="*30)
    print("Starting DQN Training Experiment...")
    print("="*30 + "\n")
    dqn_model_base_name = "dqn_robot_model"
    
    # train_dqn returns the paths to the best and final models
    dqn_best_model_path, dqn_final_model_path = train_dqn(
        num_episodes=num_episodes, 
        render_mode=render_mode, 
        model_save_path=dqn_model_base_name
    )
    print("\n" + "="*30)
    print("DQN Training Experiment Finished.")
    print("="*30 + "\n")

    # Evaluate DQN Final Model
    if dqn_final_model_path and os.path.exists(dqn_final_model_path):
        print("\n--- Evaluating DQN Final Model ---")
        evaluate_model(dqn_final_model_path, QNetwork, num_eval_episodes=5, render=True)
    else:
        print("DQN final model not found for evaluation.")

    # Evaluate DQN Best Model 
    if dqn_best_model_path and os.path.exists(dqn_best_model_path):
        print("\n--- Evaluating DQN Best Model ---")
        evaluate_model(dqn_best_model_path, QNetwork, num_eval_episodes=5, render=True)
    else:
        print("DQN best model not found for evaluation.")

    # Zip DQN Models
    if dqn_best_model_path and dqn_final_model_path:
        zip_models('dqn', 'best', [dqn_best_model_path])
        zip_models('dqn', 'final', [dqn_final_model_path])
        # If your assignment requires a single zip for both best/final for an algo:
        zip_models('dqn', 'both', [dqn_best_model_path, dqn_final_model_path])
    else:
        print("Skipping DQN model zipping: Best or Final model path missing.")


    # --- Policy Gradient Experiment (COMMENTED OUT FOR NOW) ---
    # print("\n\n" + "="*30)
    # print("Starting Policy Gradient Training Experiment...")
    # print("="*30 + "\n")
    # pg_model_base_name = "pg_robot_model"

    # pg_best_model_path, pg_final_model_path = train_pg(
    #     num_episodes=num_episodes,
    #     render_mode=render_mode,
    #     model_save_path=pg_model_base_name
    # )
    # print("\n" + "="*30)
    # print("Policy Gradient Training Experiment Finished.")
    # print("="*30 + "\n")

    # # Evaluate PG Final Model
    # if pg_final_model_path and os.path.exists(pg_final_model_path):
    #     print("\n--- Evaluating Policy Gradient Final Model ---")
    #     evaluate_model(pg_final_model_path, PolicyNetwork, num_eval_episodes=5, render=True)
    # else:
    #     print("Policy Gradient final model not found for evaluation.")

    # # Evaluate PG Best Model
    # if pg_best_model_path and os.path.exists(pg_best_model_path):
    #     print("\n--- Evaluating Policy Gradient Best Model ---")
    #     evaluate_model(pg_best_model_path, PolicyNetwork, num_eval_episodes=5, render=True)
    # else:
    #     print("Policy Gradient best model not found for evaluation.")

    # # Zip PG Models
    # if pg_best_model_path and pg_final_model_path:
    #     zip_models('pg', 'best', [pg_best_model_path])
    #     zip_models('pg', 'final', [pg_final_model_path])
    #     zip_models('pg', 'both', [pg_best_model_path, pg_final_model_path])
    # else:
    #     print("Skipping Policy Gradient model zipping: Best or Final model path missing.")


if __name__ == "__main__":
    run_dqn_experiment() # Call the DQN-only experiment function