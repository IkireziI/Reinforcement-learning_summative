import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Import the new Renderer class
from environment.rendering import RobotRenderer

class DeliveryRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    # Define constants for grid cell types (used in _grid_map)
    EMPTY = 0
    OBSTACLE = 1

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid (e.g., 10x10)
        self.window_size = 512  # The size of the PyGame window for rendering

        # --- MODIFIED: Max steps for episode truncation ---
        # Defines the maximum number of steps an agent can take in one episode
        self._max_episode_steps = self.size * self.size * 5 # Increased to 500 steps for 10x10 (previously 200)
        self._current_steps = 0 # Counter to track steps taken in the current episode
        # --- END MODIFICATION ---

        # Define the fixed warehouse grid map
        # 0: Empty, 1: Obstacle
        self._grid_map = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=int)
        assert self._grid_map.shape == (self.size, self.size), "Grid map size must match 'size' parameter."

        # Define fixed loading zones and delivery stations (coordinates must be EMPTY cells in _grid_map)
        self._loading_zones = [[0,0], [9,9]] # Top-left, Bottom-right corners
        self._delivery_stations = [[0,9], [9,0]] # Top-right, Bottom-left corners

        # Ensure chosen zones/stations are actually empty in the map for valid placement/movement
        for loc in self._loading_zones + self._delivery_stations:
            # Note: (y, x) for numpy array indexing
            assert self._grid_map[loc[1], loc[0]] == self.EMPTY, f"Zone/Station {loc} is not an EMPTY cell in the grid map!"

        # Observation Space: (robot_x, robot_y, has_package, package_x, package_y, target_x, target_y)
        # x,y coordinates from 0 to size-1
        # has_package is 0 or 1
        # package_x, package_y will be -1,-1 if robot has package (or package delivered)
        low_bounds = np.array([0, 0, 0, -1, -1, 0, 0], dtype=np.int32)
        high_bounds = np.array([self.size - 1, self.size - 1, 1, self.size - 1, self.size - 1, self.size - 1, self.size - 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int32)

        # Action Space: 0:Up, 1:Down, 2:Left, 3:Right, 4:Pick Up, 5:Drop Off
        self.action_space = spaces.Discrete(6)

        self.render_mode = render_mode

        # Initialize the renderer only if a render_mode is specified
        self.renderer = None
        if self.render_mode is not None:
            self.renderer = RobotRenderer(
                self.size,
                self.window_size,
                self._grid_map,
                self._loading_zones,
                self._delivery_stations
            )
            print("DEBUG: DeliveryRobotEnv - Renderer initialized!")

    def _get_obs(self):
        # Package location will be (-1,-1) if the robot holds it or it's delivered
        pkg_x, pkg_y = (-1, -1)
        if not self._agent_has_package and self._package_location is not None:
            pkg_x, pkg_y = self._package_location[0], self._package_location[1]

        target_x, target_y = self._target_delivery_station[0], self._target_delivery_station[1]

        return np.array([
            self._agent_location[0],
            self._agent_location[1],
            int(self._agent_has_package),
            pkg_x,
            pkg_y,
            target_x,
            target_y
        ], dtype=np.int32)

    def _get_info(self):
        info = {
            "robot_location": tuple(self._agent_location),
            "has_package": self._agent_has_package,
            "package_on_map": tuple(self._package_location) if self._package_location is not None and not self._agent_has_package else None,
            "target_station": tuple(self._target_delivery_station)
        }
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset step counter for a new episode
        self._current_steps = 0

        # 1. Initialize Robot Location
        # Find all empty cells to place the robot randomly
        valid_spawn_cells = np.argwhere(self._grid_map == self.EMPTY).tolist()
        self._agent_location = np.array(self.np_random.choice(valid_spawn_cells), dtype=int)

        # 2. Initialize Package Location and Target Delivery Station
        # Select a random loading zone for the package
        pkg_zone_loc_idx = self.np_random.choice(len(self._loading_zones))
        pkg_zone_loc = self._loading_zones[pkg_zone_loc_idx]
        self._package_location = np.array(pkg_zone_loc, dtype=int)

        # Select a random delivery station for this package.
        # Make sure it's not the same as the package pickup zone (if possible)
        delivery_station_options = [s for s in self._delivery_stations if not np.array_equal(s, pkg_zone_loc)]
        if not delivery_station_options:
            delivery_station_options = self._delivery_stations

        del_station_loc_idx = self.np_random.choice(len(delivery_station_options))
        del_station_loc = delivery_station_options[del_station_loc_idx]
        self._target_delivery_station = np.array(del_station_loc, dtype=int)


        self._agent_has_package = False
        self._package_delivered = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.renderer:
            self.renderer.render_frame(
                self._agent_location,
                self._agent_has_package,
                self._package_location,
                self._target_delivery_station,
                self.metadata["render_fps"]
            )

        return observation, info

    def step(self, action):
        self._current_steps += 1 # Increment step counter for each action
        reward = -0.1 # Small penalty for each step
        terminated = False
        truncated = False

        current_pos = self._agent_location.copy()

        # Handle Movement Actions (0:Up, 1:Down, 2:Left, 3:Right)
        if action == 0:  # Up (decrease y)
            next_pos = current_pos + np.array([0, -1])
        elif action == 1:  # Down (increase y)
            next_pos = current_pos + np.array([0, 1])
        elif action == 2:  # Left (decrease x)
            next_pos = current_pos + np.array([-1, 0])
        elif action == 3:  # Right (increase x)
            next_pos = current_pos + np.array([1, 0])
        else:
            next_pos = current_pos # For pick/drop actions, position doesn't change

        # Check for boundary and obstacle collisions for movement actions (actions 0-3)
        if action < 4:
            # Check boundaries first
            if (0 <= next_pos[0] < self.size and
                0 <= next_pos[1] < self.size):
                # Check for obstacle
                if self._grid_map[next_pos[1], next_pos[0]] != self.OBSTACLE:
                    self._agent_location = next_pos # Valid move
                else:
                    reward -= 5 # Collision with obstacle
            else:
                reward -= 5 # Collision with boundary

        # Handle Pick Up Action (action 4)
        elif action == 4: # Pick Up
            if not self._agent_has_package:
                # Check if robot is at package location
                if self._package_location is not None and np.array_equal(self._agent_location, self._package_location):
                    self._agent_has_package = True
                    self._package_location = None # Package is now with the robot
                    reward += 20 # Positive reward for picking up
                else:
                    reward -= 5 # Penalty for trying to pick up nothing or wrong place
            else:
                reward -= 1 # Small penalty for trying to pick up when already has package

        # Handle Drop Off Action (action 5)
        elif action == 5: # Drop Off
            if self._agent_has_package:
                # Check if robot is at the target delivery station
                if np.array_equal(self._agent_location, self._target_delivery_station):
                    self._agent_has_package = False
                    # --- MODIFIED: Increased reward for successful delivery ---
                    reward += 1000 # Increased from 100 to 1000
                    # --- END MODIFICATION ---
                    terminated = True
                else:
                    reward -= 5 # Penalty for dropping off at wrong location
            else:
                reward -= 1 # Small penalty for trying to drop off when no package

        # Handle Truncation: Check if the maximum number of steps for this episode has been reached
        if self._current_steps >= self._max_episode_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        # Render logic
        if self.render_mode == "rgb_array" and self.renderer:
            frame = self.renderer.render_frame(
                self._agent_location,
                self._agent_has_package,
                self._package_location,
                self._target_delivery_station,
                self.metadata["render_fps"]
            )
            return observation, reward, terminated, truncated, info, frame
        elif self.render_mode == "human" and self.renderer:
            self.renderer.render_frame(
                self._agent_location,
                self._agent_has_package,
                self._package_location,
                self._target_delivery_station,
                self.metadata["render_fps"]
            )
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.renderer:
            return self.renderer.render_frame(
                self._agent_location,
                self._agent_has_package,
                self._package_location,
                self._target_delivery_station,
                self.metadata["render_fps"]
            )
        return None

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None