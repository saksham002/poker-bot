# Gemini Project Context: Poker Bot

This document provides context for the Gemini AI assistant to effectively assist with this project.

## 1. Project Goal

This project is a command-line Texas Hold'em poker game. It features human players and an AI-powered `AutomatedPlayer` that uses a reinforcement learning approach (Policy/Critic networks) to make decisions.

## 2. Key Technologies

- **Language:** Python 3.12
- **Core Libraries:**
    - `torch`: For the neural networks (Policy and Critic).
    - `treys`: For poker hand evaluation.

## 3. Project Structure

- `main.py`: The main entry point for starting the game. Handles user input for game setup.
- `game.py`: Contains the core game logic, including managing rounds, betting, and determining winners.
- `player.py`: Defines the base `Player` class with common attributes and actions.
- `automated_player.py`: Defines the `AutomatedPlayer` class, which contains the AI logic, neural networks, and training methods.
- `checkpoints/`: Directory where trained model checkpoints are saved.

## 4. Development Workflow

### How to Run the Game
To run the application, execute the following command from the project root:
```bash
python main.py
```

### How to Run Tests
There are currently no automated tests for this project. Assistance in creating a test suite using `pytest` would be valuable.

## 5. Coding Conventions & Style

- Follow the PEP 8 style guide.
- Use type hints for all new function definitions.
- The AI model's state vector in `automated_player.py` is sensitive. It's composed of `[num_cards_shown, win_probabilities, other_player_bet_ratios]`. Be mindful of this structure when making changes to the AI.

## 6. TODO's
- Assumes num_players does not change, fix this.
- Approach is player-style agnostic, potentially would like to add a state vector that describes play-style in some abstract space that is learnt for each player starting with a default rational value.
- No option for the AI agent to raise during a betting round.
- Not only should the num_cards_shown variable be used as input to the NNs but also some encoding of the cards on the table revealed thus far.

## 7. Prompt Conventions
When a file/directory name is mentioned using "@", any numbers following the file name (after an additional space) are meant to represent line numbers in the following formats:
- @<file_name> <number> : This will be used to refer to just a single line.
- @<file_name> <number_1>,<number_2>,...,<number_k> : This will be used to refer to several line numbers comma-separated.
- @<file_name> <number_l>-<number_r> : This will be used to refer to a range of lines inclusive of both ends.
- The above two rules can be applied simultaneously as well to refer to a combination of stand-alone lines and ranges of lines.
