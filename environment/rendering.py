import pygame
import numpy as np

class RobotRenderer:
    def __init__(self, size, window_size, grid_map, loading_zones, delivery_stations):
        self.size = size
        self.window_size = window_size
        self.grid_map = grid_map
        self.loading_zones = loading_zones
        self.delivery_stations = delivery_stations

        # Define constants for grid cell types (match those in custom_env.py)
        self.EMPTY = 0
        self.OBSTACLE = 1

        # Define colors for rendering
        self.COLORS = {
            self.EMPTY: (255, 255, 255),  # White
            self.OBSTACLE: (100, 100, 100), # Gray
            "robot": (255, 0, 0),    # Red
            "package": (255, 255, 0), # Yellow
            "grid_lines": (0, 0, 0) # Black
        }
        # Specific colors for drawing loading zones and delivery stations as highlights
        self.LOADING_ZONE_COLOR = (0, 200, 0)  # Green
        self.DELIVERY_STATION_COLOR = (0, 0, 200) # Dark Blue
        self.TARGET_MARKER_COLOR = (255, 0, 255) # Magenta for current target
        self.PACKAGE_ON_ROBOT_COLOR = (0, 150, 255) # Lighter Blue for package on robot

        self.window = None
        self.clock = None

    def _init_pygame(self):
        # NEW DEBUG LINE: Check the state of self.window right at the start
        print(f"DEBUG: RobotRenderer - _init_pygame called. self.window is currently: {self.window}")
        
        # This condition ensures Pygame initialization (and window creation)
        # only happens once per renderer instance.
        if self.window is None:
            pygame.init() # Initializes all Pygame modules
            pygame.display.init() # Initializes the display module specifically
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            print("DEBUG: RobotRenderer - Pygame window created!") # This indicates successful window creation
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render_frame(self, agent_location, agent_has_package, package_location, target_delivery_station, fps):
        print("DEBUG: RobotRenderer - render_frame called!")
        self._init_pygame() # Ensure Pygame window is initialized

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLORS[self.EMPTY])  # Default white background

        pix_square_size = self.window_size / self.size

        # Draw grid cells based on _grid_map
        for y in range(self.size):
            for x in range(self.size):
                cell_type = self.grid_map[y, x]
                color = self.COLORS.get(cell_type, self.COLORS[self.EMPTY])
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        x * pix_square_size,
                        y * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )

        # Highlight all Loading Zones (as outlines)
        for lz_x, lz_y in self.loading_zones:
            pygame.draw.rect(
                canvas,
                self.LOADING_ZONE_COLOR,
                pygame.Rect(
                    lz_x * pix_square_size,
                    lz_y * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
                width=2 # Draw as an outline
            )

        # Highlight all Delivery Stations (as outlines)
        for ds_x, ds_y in self.delivery_stations:
            pygame.draw.rect(
                canvas,
                self.DELIVERY_STATION_COLOR,
                pygame.Rect(
                    ds_x * pix_square_size,
                    ds_y * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
                width=2 # Draw as an outline
            )


        # Draw current package location if it's on the map and not on the agent
        if package_location is not None and not agent_has_package:
            pygame.draw.circle(
                canvas,
                self.COLORS["package"], # Yellow
                (
                    (package_location[0] + 0.5) * pix_square_size,
                    (package_location[1] + 0.5) * pix_square_size,
                ),
                pix_square_size / 4, # Smaller circle for package
            )
        
        # Draw the target delivery station marker (magenta outline circle)
        # This helps to clearly show which delivery station is the current target
        pygame.draw.circle(
            canvas,
            self.TARGET_MARKER_COLOR,
            (
                (target_delivery_station[0] + 0.5) * pix_square_size,
                (target_delivery_station[1] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3, # Larger circle for target station
            width=2 # Make it an outline
        )


        # Draw the agent (robot)
        pygame.draw.circle(
            canvas,
            self.COLORS["robot"], # Red
            (
                (agent_location[0] + 0.5) * pix_square_size,
                (agent_location[1] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

        # Draw a smaller blue circle on the robot if it has a package
        if agent_has_package:
            pygame.draw.circle(
                canvas,
                self.PACKAGE_ON_ROBOT_COLOR, # Lighter Blue
                (
                    (agent_location[0] + 0.5) * pix_square_size,
                    (agent_location[1] + 0.5) * pix_square_size,
                ),
                pix_square_size / 6, # Even smaller circle
            )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                self.COLORS["grid_lines"], # Black
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
            pygame.draw.line(
                canvas,
                self.COLORS["grid_lines"], # Black
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )

        # --- THIS IS THE CRUCIAL NEW LINE TO BLIT THE CANVAS TO THE DISPLAY ---
        self.window.blit(canvas, (0, 0)) 

        pygame.event.pump() # Process Pygame events
        pygame.display.update() # Update the display
        self.clock.tick(fps) # Control frame rate
        
        # Return the pixel data if needed (e.g., for rgb_array render mode)
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # Reset window after closing
            self.clock = None # Reset clock