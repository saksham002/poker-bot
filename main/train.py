import os
from tqdm import tqdm
import sys
from base_classes.game import Game
from base_classes.plotter import Plotter

if __name__ == "__main__":
    num_players, buy_in, min_bet, num_games = map(int, input("Enter number of players, buy in, minimum bet and number of games to simulate: ").split())
    load_checkpt_policy, load_checkpt_critic, lmbda = ["" for i in range(num_players)], ["" for i in range(num_players)], 0.0

    policy_paths = input(f"Enter {num_players} policy network paths to load from (separated by spaces): ").split()
    assert len(policy_paths) == num_players, f"Expected {num_players} policy paths, but received {len(policy_paths)}"
    critic_paths = input(f"Enter {num_players} critic network paths to load from (separated by spaces): ").split()
    assert len(critic_paths) == num_players, f"Expected {num_players} critic paths, but received {len(critic_paths)}"

    for i in range(num_players):
        if os.path.exists(policy_paths[i]):
            load_checkpt_policy[i] = policy_paths[i]
        if os.path.exists(critic_paths[i]):
            load_checkpt_critic[i] = critic_paths[i]
    
    lmbda = float(input("Enter value of hyper-parameter lambda to train the network: ")) 

    plotter = Plotter([f"P{i}" for i in range(num_players)])
    player_data = [[1e-3, 0] for i in range(num_players)]

    os.makedirs(f"train_out", exist_ok = True)
    log_file_path = f"train_out/{num_players}.txt"

    original_stdout = sys.stdout
    try:
        with open(log_file_path, "w") as f:
            sys.stdout = f
            for i in tqdm(range(num_games), desc = "Training Progress"):
                print(f"----------------------------Game {i + 1}: Start----------------------------")
                game = Game(num_players, buy_in, min_bet, False, True, load_checkpt_policy, load_checkpt_critic, True, lmbda, plotter, i % 2, player_data)
                game.game_num = i
                ctr = 1
                while game.num_players == num_players:
                    print(f"--------------Round {ctr}: Start--------------")
                    game.round()
                    print(f"--------------Round {ctr}: End--------------")
                    ctr += 1
                load_checkpt_policy, load_checkpt_critic, player_data = game.end()
                print(f"----------------------------Game {i + 1}: End----------------------------")
                print(load_checkpt_policy, load_checkpt_critic, sep = " ")
    finally:
        sys.stdout = original_stdout
        plotter.close()
