# Reinforcement-learning_summative

## Project Description ##
This project demonstrates a reinforcement learning model in a grid-based Pygame environment. The goal is to train an agent to navigate a grid and perform a specific task.

## How the Environment Works ##
The Pygame window displays a grid with several key elements:

The Red Ball: This represents the agent (or player). Its objective is to learn how to move through the grid to achieve a goal.

The Yellow Ball: This is the target delivery station, or the final destination for the agent. The agent must reach this point to successfully complete an episode.

The Pink/Purple Circle: This represents the package. The agent's task is to first pick up this package before heading to the yellow ball.

White Squares: These are the valid spaces the agent can move into.

Grey Squares: These are obstacles or walls. The agent cannot move into these spaces and must learn to navigate around them to reach its destination.

The Blue Square: This is the starting position of the agent at the beginning of each episode. The red ball will spawn here at the start of every game.

## Reinforcement Learning in Action ##
The game is not played manually by a person. Instead, a reinforcement learning agent (either a DQNAgent or PGAgent, as seen in your code and terminal output) learns to play the game on its own.

## The agent learns by receiving rewards and penalties: ## 

The agent is rewarded for moving toward the package and the delivery station.

The agent receives a penalty for colliding with a wall or taking too long to complete the task

## Setup and Installation ##
Follow these steps to set up the project on your local machine.

## Prerequisites ##
Python: Ensure you have Python installed (version 3.8 or newer is recommended).

Git: Make sure Git is installed to clone the repository.

Step 1: Clone the Repository

Open your terminal or command prompt and run the following command to download the project files from GitHub.

git clone https://github.com/IkireziI/Reinforcement-learning_summative.git
cd Reinforcement-learning_summative

Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage the project's dependencies.

On Windows:

python -m venv venv
.\venv\Scripts\activate

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

Step 3: Install Required Libraries

With your virtual environment activated, install all the necessary Python libraries from the requirements.txt file.

pip install -r requirements.txt

Step 4: Run the Project

After all the dependencies are installed, you can start the project by running the main Python script.

python main.py

This will begin the training and evaluation process for both the DQNAgent and the PGAgent in the Pygame environment.




Query successful
Reinforcement Learning Project
Project Description
This project demonstrates a reinforcement learning model in a grid-based Pygame environment. The goal is to train an agent to navigate a grid and perform a specific task.

The Pygame window displays a grid with several key elements:

The Red Ball: This represents the agent (or player). Its objective is to learn how to move through the grid to achieve a goal.

The Yellow Ball: This is the target delivery station, or the final destination for the agent. The agent must reach this point to successfully complete an episode.

The Pink/Purple Circle: This represents the package. The agent's task is to first pick up this package before heading to the yellow ball.

White Squares: These are the valid spaces the agent can move into.

Grey Squares: These are obstacles or walls. The agent cannot move into these spaces and must learn to navigate around them to reach its destination.

The Blue Square: This is the starting position of the agent at the beginning of each episode. The red ball will spawn here at the start of every game.

The game is not played manually by a person. Instead, a reinforcement learning agent (either a DQNAgent or PGAgent) learns to play the game on its own. The agent learns by receiving rewards and penalties. Over many episodes, the agent's goal is to learn the optimal path to follow in the environment to maximize its total reward, eventually becoming an expert at navigating the grid to pick up the package and deliver it to the target delivery station.

Setup and Installation
Follow these steps to set up the project on your local machine.

Prerequisites
Python: Ensure you have Python installed (version 3.8 or newer is recommended).

Git: Make sure Git is installed to clone the repository.

Step 1: Clone the Repository
Open your terminal or command prompt and run the following command to download the project files from GitHub.

Bash

git clone https://github.com/IkireziI/Reinforcement-learning_summative.git
cd Reinforcement-learning_summative