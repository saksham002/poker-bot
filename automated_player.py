import time
from treys import Card, Deck, Evaluator
from player import Player
from generic_helper import debug_print
from math import exp
import random
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

init_scale = random.uniform(0,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype = torch.float32))

    def forward(self, x, coef, b):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        win_probs = F.softmax(self.fc3(x), dim = 1)
        y = torch.sum(win_probs * coef, dim = -1) + b
        log_prob_call = F.logsigmoid(self.scale * y)
        return win_probs, log_prob_call

class CriticNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# for now, just decide whether to fold or call
class AutomatedPlayer(Player):
    def __init__(self, name, buy_in, num_players, min_bet, load_checkpt_policy = "", load_checkpt_critic = "", train_network = False, lmbda = 0.1):
        # to-do list
        # 1. assumes num_players does not change, fix this.
        # 2. approach is player-style agnostic and does not give the program an option to raise.
        # 3. encodes number of cards shown but not the cards themselves in the state vector.

        super().__init__(name, buy_in, num_players, min_bet)
        self.lmbda = lmbda
        self.num_updates = 0
        self.table_cards_so_far = []
        self.num_players_round = num_players
        self.pot = 0
        self.policy_nn = PolicyNN(2 * num_players, 8, num_players + 1).to(device)
        if load_checkpt_policy != "":
            self.policy_nn.load_state_dict(torch.load(load_checkpt_policy, map_location = device))
        self.critic_nn = CriticNN(2 * num_players, 8).to(device)
        if load_checkpt_critic != "":
            self.critic_nn.load_state_dict(torch.load(load_checkpt_critic, map_location = device))
        self.train_network = train_network
        if self.train_network:
            self.policy_optimizer = torch.optim.Adam(self.policy_nn.parameters(), lr = 1e-3)
            self.critic_optimizer = torch.optim.Adam(self.critic_nn.parameters(), lr = 1e-3)
        else:
            self.policy_nn.eval()
            self.critic_nn.eval()
        self.round_action_dict = {"self_round_bets" : [], "other_players_round_bets" : [], "other_players_money" : [], "pots" : [], "action_regrets": []}
        self.nn_vals = {"num_folds" : [], "state_vecs" : [], "win_probs" : [], "log_prob_action" : [], "critic_outputs" : []}

    def update_num_players(self, new_val):
        if self.num_players != new_val:
            raise ValueError("Cannot decrement num_players with current approach.")
        self.num_players = new_val

    def decrement_num_players_round(self):
        self.num_players_round -= 1

    def reset(self):
        self._hand = [] 
        self.active = True
        self.round_bet = 0
        self.max_bet = 0
        self.table_cards_so_far = []
        self.num_players_round = self.num_players
        self.pot = 0
        self.round_action_dict = {"self_round_bets" : [], "other_players_round_bets" : [], "other_players_money" : [], "pots" : [], "action_regrets": []}
        self.nn_vals = {"num_folds" : [], "state_vecs" : [], "win_probs" : [], "log_prob_action" : [], "critic_outputs" : []}

    def set_max_bet(self, max_bet, pot):
        self.pot = pot
        self.max_bet = max_bet

    def update_action_dict(self, round_bet, money):
        if len(self.round_action_dict["other_players_round_bets"]) == 0 or len(self.round_action_dict["other_players_round_bets"][-1]) == self.num_players - 1:
            self.round_action_dict["other_players_round_bets"].append([])
            self.round_action_dict["other_players_money"].append([])
        self.round_action_dict["other_players_round_bets"][-1].append(round_bet)
        self.round_action_dict["other_players_money"][-1].append(money + round_bet)

    def mc_split_distribution(self, board, k, num_trials = 20000):
        """
        Monte-Carlo estimate of:
        P(win alone), P(split among exactly 2 players), …, P(split among k players)
        Returns a list of length k: P[0..k-1] as described above.
        """

        self_cards = [Card.new(c) for c in self._hand]
        known_board = [Card.new(c) for c in board]
        evaluator = Evaluator()
        
        # Prepare the counts
        counts = [0] * k
        
        for _ in range(num_trials):
            # build & shuffle fresh deck
            deck = Deck()
            # remove known cards
            for c in self_cards + known_board:
                deck.cards.remove(c)
            
            # fill out the board to 5 cards
            board_extra = deck.draw(5 - len(known_board))
            full_board = known_board + board_extra
            
            # evaluate score
            self_score = evaluator.evaluate(full_board, self_cards)
            
            # deal opponents
            opp_scores = []
            for _ in range(k - 1):
                opp = deck.draw(2)
                opp_scores.append(evaluator.evaluate(full_board, opp))
            
            # determine winners
            all_scores = [self_score] + opp_scores
            best = min(all_scores)         # lower = stronger in treys
            winners = [i for i, s in enumerate(all_scores) if s == best]
            
            # if self is one of the winners, record the split size
            if 0 in winners:
                split_size = len(winners)  # how many share the pot
                counts[split_size - 1] += 1
        
        # normalize to probabilities
        # self.show_hand()
        probs = [cnt / num_trials for cnt in counts]
        # print(probs)
        return probs

    def expected_value_call(self):
        k = self.num_players_round
        probs = self.mc_split_distribution(self.table_cards_so_far, k)
        b = self.round_bet
        B = min([self.money + self.round_bet, self.max_bet])
        P = self.pot
        E = b
        for i in range(self.num_players_round):
            E += ((P + B - b) / (i + 1) - B) * probs[i]
        E += -1 * B * (1 - sum(probs))
        # print(E)
        return E

    def compute_action_regrets(self, num_winners, is_winner, winning_score, board):
        action_regrets = []
        bets = self.round_action_dict["self_round_bets"]
        pots = self.round_action_dict["pots"]

        final_pot = self.pot

        if is_winner:
            # Player is one of the winners
            if final_pot >= self.round_bet:
                self.round_action_dict["action_regrets"] = [0] * len(bets)
                return
            
            for i in range(len(pots)):
                p_prev = pots[i - 1] if i > 0 else 0
                p_curr = pots[i] if i < len(pots) - 1 else final_pot
                regret = (p_curr - p_prev) / num_winners - bets[i]
                action_regrets.append(regret)
        else:
            # Player folded
            evaluator = Evaluator()
            player_score = evaluator.evaluate([Card.new(c) for c in board], [Card.new(c) for c in self._hand])
            
            num_winners_prime = 0
            if player_score < winning_score: # Hand was better
                num_winners_prime = 1
            elif player_score == winning_score: # Hand was a tie
                num_winners_prime = num_winners + 1

            if num_winners_prime > 0:
                # Calculate regrets for the actions that were calls
                for i in range(len(pots) - 1):
                    p_prev = pots[i - 1] if i > 0 else 0
                    regret = (pots[i] - p_prev) / num_winners_prime - bets[i]
                    action_regrets.append(regret)
                
                # Calculate regret for the fold action
                money_before_hand = self.money + self.round_bet
                fold_regret = min(money_before_hand, self.max_bet) - (final_pot - (pots[-2] if len(pots) > 1 else 0)) / num_winners_prime
                action_regrets.append(fold_regret)
            else:
                # Hand was worse, original logic
                for bet in bets:
                    action_regrets.append(-bet)

        self.round_action_dict["action_regrets"] = action_regrets

    def cue_for_action(self):
#        start_time = time.time()  # time in seconds (float)
    
        old_round_bet = self.round_bet

        if self.money == 0 or self.round_bet == self.max_bet:
            self.check()
        else:
            num_cards_shown = len(self.table_cards_so_far)    
            rbs, tots = self.round_action_dict["other_players_round_bets"][-1], self.round_action_dict["other_players_money"][-1] 
            add_bets = []
            k = len(rbs)

            for i in range(k):
                if rbs[i] == 0 and tots[i] > 0:
                    add_bets.append(-1)
                elif rbs[i] == 0:
                    add_bets.append(1)
                else:
                    add_bets.append(rbs[i] / tots[i])

            orig_probs = torch.tensor(self.mc_split_distribution(self.table_cards_so_far, self.num_players), device = device).unsqueeze(0)
            state_vec = torch.cat((torch.tensor([num_cards_shown], device = device).unsqueeze(0), orig_probs, torch.tensor(add_bets, device = device).unsqueeze(0)), dim = 1)
            self.nn_vals["state_vecs"].append(state_vec)

            folds = add_bets.count(-1)
            self.nn_vals["num_folds"].append(folds)
        
            b = self.round_bet
            B = min([self.money + self.round_bet, self.max_bet])
            P = self.pot
            M = [(P + B - b) / (i + 1) - B for i in range(self.num_players)]
            M.append(-B)
            M = torch.tensor(M, device = device).unsqueeze(0)
            b = torch.tensor([b], device = device)
            mod_probs, log_prob_call = self.policy_nn(state_vec, M, b)

            if random.random() < exp(log_prob_call):
                self.nn_vals["log_prob_action"].append(log_prob_call)
                try:
                    self.call()
                except ValueError:
                    self.max_bet = self.all_in()
            else:
                self.nn_vals["log_prob_action"].append(torch.log1p(-torch.exp(log_prob_call)))
                self.fold()

            self.round_action_dict["self_round_bets"].append(self.round_bet - old_round_bet)
            self.round_action_dict["pots"].append(P + self.round_bet - old_round_bet)

            self.nn_vals["win_probs"].append(mod_probs)

            V_t = self.critic_nn(state_vec)
            self.nn_vals["critic_outputs"].append(V_t)

        add_to_pot = self.round_bet - old_round_bet

#        end_time = time.time()
#        elapsed_ms = (end_time - start_time) * 1000  # convert to milliseconds
        # print(f"cue_for_action took {elapsed_ms:.2f} ms to run.")

        return add_to_pot, self.max_bet

    def train_iter(self):
        if len(self.round_action_dict["action_regrets"]) == 0:
            return

        rewards = self.round_action_dict["action_regrets"]
        log_probs = torch.cat(self.nn_vals["log_prob_action"], dim = 0)
        critic_values = torch.cat(self.nn_vals["critic_outputs"], dim = 0)
        win_probs = torch.cat(self.nn_vals["win_probs"], dim = 0)
        num_folds = self.nn_vals["num_folds"]

        # Calculate advantages
        advantages = []
        with torch.no_grad():
            for t in range(len(rewards)):
                v_t = critic_values[t]
                v_t_plus_1 = critic_values[t + 1] if t < len(rewards) - 1 else 0
                advantage = rewards[t] + v_t_plus_1 - v_t
                advantages.append(advantage)
        advantages = torch.cat(advantages).to(dtype = torch.float32)

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Fold loss
        fold_loss = 0
        k = self.num_players
        for i in range(len(win_probs)):
            fold_loss += torch.sum(torch.abs(win_probs[i][k - num_folds[i] : k]))
        fold_loss /= len(win_probs)

        # Total policy loss
        total_policy_loss = policy_loss + self.lmbda * fold_loss

        # Critic loss
        suffix_rewards = []
        for i in range(len(rewards)):
            suffix_rewards.append(sum(rewards[i : ]))
        critic_loss = F.mse_loss(torch.tensor(suffix_rewards, device = device, dtype = torch.float32), critic_values.squeeze(-1))

        # debug_print(f"rewards: {rewards}\nlog_probs: {log_probs}\ncritic_values: {critic_values}\nwin_probs: {win_probs}\nnum_folds: {num_folds}\nadvantages: {advantages}\nsuffix_rewards: {suffix_rewards}")
        # debug_print(f"critic_loss: {critic_loss}\npolicy_loss: {policy_loss}\nfold_loss: {fold_loss}")

        # Backpropagation
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.num_updates += 1
        if self.num_updates % 50 == 0:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            policy_checkpoint_path = f"checkpoints/policy/policy_checkpoint_{now}_{self.num_updates}_{self.num_players}.pth"
            critic_checkpoint_path = f"checkpoints/critic/critic_checkpoint_{now}_{self.num_updates}_{self.num_players}.pth"
            os.makedirs(os.path.dirname(policy_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(critic_checkpoint_path), exist_ok=True)
            torch.save(self.policy_nn.state_dict(), policy_checkpoint_path)
            torch.save(self.critic_nn.state_dict(), critic_checkpoint_path)
