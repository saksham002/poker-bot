# Poker Bot

## Requirements

The following libraries are required to run this project:

*   `torch`
*   `treys`

You can install them using pip:
```bash
pip install torch treys
```

## Setup

Before running the project, you need to set the `PYTHONPATH` to the project's root directory. You can do this by running the following command from the project's root directory:

```bash
export PYTHONPATH=$(pwd)
```

## Training Observation

An important observation from training the model is that flipping the small blind player for the first round of every game results in a better solution. This is implemented in `main/train.py`.
