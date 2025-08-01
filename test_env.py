import gymnasium as gym
from environment.custom_env import DeliveryRobotEnv
import time

def run_test_env(num_episodes=5, max_steps_per_episode=100):
    """
    Runs a test simulation of the DeliveryRobotEnv with random actions.
    """
    print(f"Running test environment for {num_episodes} episodes.")

    # Create the environment in human render mode to display the window
    # Ensure render_mode is "human" so you can see the simulation
    env = DeliveryRobotEnv(render_mode="human", size=10)

    for episode in range(num_episodes):
        observation, info = env.reset()
        print(f"\n--- Episode {episode + 1} Started ---")
        print(f"Initial Robot Location: {info['robot_location']}")
        print(f"Initial Package Location: {info['package_on_map']}")
        print(f"Target Delivery Station: {info['target_station']}")

        total_reward = 0
        terminated = False
        truncated = False
        steps = 0

        # Render the initial frame
        env.render()
        time.sleep(0.5) # Short pause to see initial state

        while not terminated and not truncated and steps < max_steps_per_episode:
            action = env.action_space.sample() # Take a random action

            # For `human` render_mode, the step method usually returns 5 values (obs, reward, terminated, truncated, info)
            # The 6th value (frame) is only returned in `rgb_array` mode.
            # So we catch either 5 or 6 values depending on how the step method returns.
            step_result = env.step(action)
            if len(step_result) == 6:
                observation, reward, terminated, truncated, info, _ = step_result # Unpack 6, ignore frame
            else:
                observation, reward, terminated, truncated, info = step_result # Unpack 5

            total_reward += reward
            steps += 1

            env.render() # Render each step
            time.sleep(0.1) # Small delay to make movements visible

        print(f"--- Episode {episode + 1} Finished ---")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Taken: {steps}")
        if terminated:
            print("Episode terminated: Package delivered!")
        elif truncated:
            print("Episode truncated: Max steps reached.")
        else:
            print("Episode ended for unknown reason.")

    env.close() # Close the Pygame window

if __name__ == "__main__":
    run_test_env(num_episodes=3, max_steps_per_episode=200) # Run 3 episodes, max 200 steps each