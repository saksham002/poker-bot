from base_classes.game import Game
import os

if __name__ == "__main__":
    num_players, buy_in, min_bet, play_with_bot = map(int, input("Enter number of players, buy in, minimum bet and whether to play with an AutomatedPlayer (0/1 input) respectively: ").split())
    load_checkpt_policy, load_checkpt_critic, train_network, lmbda = "", "", 0, 0.0
    if play_with_bot == 1:
        load_checkpt_policy, load_checkpt_critic, train_network = map(str, input("Enter policy and critic network paths to load from and whether to train the network or not (0/1 input): ").split())
        if not os.path.exists(load_checkpt_policy):
            load_checkpt_policy = ""
        if not os.path.exists(load_checkpt_critic):
            load_checkpt_critic = ""
        if int(train_network) == 1:
            lmbda = float(input("Enter value of hyper-parameter lambda to train the network: ")) 
    game = Game(num_players, buy_in, min_bet, play_with_bot, False, load_checkpt_policy, load_checkpt_critic, train_network, lmbda)
    ctr = 1
    while game.num_players > 1 and (not play_with_bot or game.num_players == num_players):
        print(f"--------------Round {ctr}: Start--------------")
        game.round()
        print(f"--------------Round {ctr}: End--------------")
        ctr += 1
    print(f"Player {game.players[0].get_name()} wins the game.")