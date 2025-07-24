import time
from treys import Card, Deck, Evaluator
from math import exp
import random
import os
from datetime import datetime
from base_classes.player import Player
from generic_helper import debug_print

import torch
import torch.nn as nn
import torch.nn.functional as F

init_scale = random.uniform(0,1)
EPS = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5a = nn.Linear(hidden_dim, output_dim)
        self.fc5b = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        action_probs = F.softmax(self.fc5a(x), dim = 1)
        db_param = F.sigmoid(self.fc5b(x))
        return action_probs, db_param

class CriticNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# for now, just decide whether to fold or call
class AutomatedPlayer(Player):
    def __init__(self, name, buy_in, num_players, min_bet, load_checkpt_policy = "", load_checkpt_critic = "", train_network = False, lmbda = 0.1, player_data_entry = None):
        # to-do list
        # 1. assumes num_players does not change, fix this.
        # 2. approach is player-style agnostic and does not give the program an option to raise.
        # 3. encodes number of cards shown but not the cards themselves in the state vector.

        super().__init__(name, buy_in, num_players, min_bet)
        self.lmbda = lmbda
        self.gamma = 0.99 # Discount factor for future rewards
        if player_data_entry:
            initial_lr = player_data_entry[0]
            self.num_updates = player_data_entry[1]
        else:
            initial_lr = 1e-3
            self.num_updates = 0
        self.is_latest_action_raise = False
        self.table_cards_so_far = []
        self.original_num_players = num_players
        self.num_players_round = num_players
        self.pot = 0
        self.max_bet_before_raise = 0
        self.policy_nn = PolicyNN(3 + 12 * num_players, 128).to(device)
        if load_checkpt_policy != "":
            self.policy_nn.load_state_dict(torch.load(load_checkpt_policy, weights_only = True, map_location = device))
        self.critic_nn = CriticNN(3 + 12 * num_players, 128).to(device)
        if load_checkpt_critic != "":
            self.critic_nn.load_state_dict(torch.load(load_checkpt_critic, weights_only = True, map_location = device))
        self.train_network = train_network
        if self.train_network:
            self.policy_optimizer = torch.optim.Adam(self.policy_nn.parameters(), lr = initial_lr, weight_decay = 1e-4)
            self.critic_optimizer = torch.optim.Adam(self.critic_nn.parameters(), lr = initial_lr, weight_decay = 1e-4)

            # Explicitly set 'initial_lr' for the schedulers
            for param_group in self.policy_optimizer.param_groups:
                param_group.setdefault('initial_lr', param_group['lr'])
            for param_group in self.critic_optimizer.param_groups:
                param_group.setdefault('initial_lr', param_group['lr'])

            self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size = 1000, gamma = 0.5, last_epoch = self.num_updates - 1)
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size = 1000, gamma = 0.5, last_epoch = self.num_updates - 1)
            self.policy_nn.train()
            self.critic_nn.train()
        else:
            self.policy_nn.eval()
            self.critic_nn.eval()
        self.round_action_dict = {"self_round_bets" : [], "other_players_round_bets" : [], "other_players_total_round_bets" : [], "other_players_money" : [], "has_folded" : [], "pots" : [], "neg_action_regrets" : [], "num_cards_seen_at_action" : [], "other_players_board_card_encodings": [], "raise_rewards" : []}
        self.nn_vals = {"state_vecs" : [], "action_probs" : [], "log_prob_action" : [], "critic_outputs" : [], "expected_fold_prob_zero" : []}
        self.trajectory_buffer = [] # To store trajectories for batch updates
        self.plot_data_game = {"critic_losses" : [], "policy_losses" : [], "fold_losses" : [], "total_policy_losses" : [], "rewards" : [], "money_values": [], "policy_grad_norms": [], "critic_grad_norms": [], "batch_sizes": []}
        self.num_table_cards_since_cache = -1
        self.cached_probs = [0 for i in range(self.num_players)]

    def _encode_cards(self, cards, fill = True):
        """
        Encodes a list of up to 5 card strings into a normalized vector.
        - Suit: h,d,c,s -> 0,1,2,3 (normalized by /3)
        - Rank: 2-A -> 0-12 (normalized by /12)
        - Unrevealed cards are represented by -1.
        """
        suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        
        encoded_vector = []
        for i in range(5):
            if i < len(cards):
                card = cards[i]
                rank = rank_map[card[0]] / 12.0
                suit = suit_map[card[1]] / 3.0
                encoded_vector.extend([rank, suit])
            else:
                if fill:
                    encoded_vector.extend([-1, -1])
                else:
                    break
        return encoded_vector
        
    def update_num_players(self, new_val):
        self.num_players = new_val

    def decrement_num_players_round(self):
        self.num_players_round -= 1

    def reset(self):
        self._hand = [] 
        self.active = True
        self.is_latest_action_raise = False
        self.round_bet = 0
        self.max_bet = 0
        self.max_bet_before_raise = 0
        self.table_cards_so_far = []
        self.num_players_round = self.num_players
        self.pot = 0
        self.round_action_dict = {"self_round_bets" : [], "other_players_round_bets" : [], "other_players_money" : [], "has_folded" : [], "pots" : [], "neg_action_regrets" : [], "num_cards_seen_at_action" : [], "other_players_board_card_encodings": [], "raise_rewards" : []}
        self.round_action_dict = {"self_round_bets" : [], "other_players_round_bets" : [], "other_players_total_round_bets" : [], "other_players_money" : [], "has_folded" : [], "pots" : [], "neg_action_regrets" : [], "num_cards_seen_at_action" : [], "other_players_board_card_encodings": [], "raise_rewards" : []}
        self.nn_vals = {"state_vecs" : [], "action_probs": [], "log_prob_action" : [], "critic_outputs" : [], "expected_fold_prob_zero" : []}
        self.num_table_cards_since_cache = -1
        self.cached_probs = [0 for i in range(self.num_players)]

    def set_max_bet(self, max_bet, pot):
        self.pot = pot
        self.max_bet = max_bet

    def update_table_cards(self, new_val):
        self.table_cards_so_far = new_val

    def update_action_dict(self, round_bet, total_round_bet, money, folded, num_cards_seen, add_to_raise_reward = 0):
        if len(self.round_action_dict["other_players_round_bets"]) == 0 or len(self.round_action_dict["other_players_round_bets"][-1]) == self.num_players - 1:
            self.round_action_dict["other_players_round_bets"].append([])
            self.round_action_dict["other_players_total_round_bets"].append([])
            self.round_action_dict["other_players_money"].append([])
            self.round_action_dict["has_folded"].append([])
            self.round_action_dict["num_cards_seen_at_action"].append([])
            self.round_action_dict["other_players_board_card_encodings"].append([])
        self.round_action_dict["other_players_round_bets"][-1].append(round_bet)
        self.round_action_dict["other_players_total_round_bets"][-1].append(total_round_bet)
        self.round_action_dict["other_players_money"][-1].append(money + round_bet)
        self.round_action_dict["has_folded"][-1].append(folded)
        self.round_action_dict["num_cards_seen_at_action"][-1].append(num_cards_seen)
        self.round_action_dict["other_players_board_card_encodings"][-1].append(self._encode_cards(self.table_cards_so_far))
        if len(self.round_action_dict["raise_rewards"]) > 0:
            self.round_action_dict["raise_rewards"][-1] += add_to_raise_reward

    def mc_split_distribution(self, board, k, num_trials = 20000):
        """
        Monte-Carlo estimate of:
        P(win alone), P(split among exactly 2 players), ..., P(split among k players)
        Returns a list of length k: P[0..k - 1] as described above.
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
        self.cached_probs = probs
        self.num_table_cards_since_cache = len(self.table_cards_so_far)
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
        neg_action_regrets = []
        bets = self.round_action_dict["self_round_bets"]
        pots = self.round_action_dict["pots"]
        raise_rewards = self.round_action_dict["raise_rewards"]

        final_pot = self.pot
        money_before_hand = self.money + self.round_bet

        #debug_print(f"bets: {bets}")
        #debug_print(f"pots: {pots}")
        #debug_print(f"raise_rewards: {raise_rewards}, num_winners: {num_winners}, is_winner: {is_winner}")

        if is_winner:
            # Player is one of the winners
            for i in range(len(pots)):
                p_prev = pots[i - 1] if i > 0 else 0
                raise_reward_prev = raise_rewards[i - 1] if i > 0 else 0
                regret = (pots[i] - p_prev + raise_rewards[i] - raise_reward_prev) / num_winners - bets[i]
                neg_action_regrets.append(regret)
        elif self.is_active():
            for bet in bets:
                neg_action_regrets.append(-bet)
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
                    raise_reward_prev = raise_rewards[i - 1] if i > 0 else 0
                    regret = (pots[i] - p_prev + raise_rewards[i] - raise_reward_prev) / num_winners_prime - bets[i]
                    neg_action_regrets.append(regret)
                
                # Calculate regret for the fold action
                fold_regret = -1 * sum(bets)
                neg_action_regrets.append(fold_regret)
            else:
                # Hand was worse, original logic
                for bet in bets:
                    neg_action_regrets.append(-bet)
        
        neg_action_regrets = [neg_action_regret / self.buy_in for neg_action_regret in neg_action_regrets]

        #debug_print(f"neg_action_regrets: {neg_action_regrets}")
        self.round_action_dict["neg_action_regrets"] = neg_action_regrets

    def cue_for_action(self):
#        start_time = time.time()  # time in seconds (float)
        self.is_latest_action_raise = False

        if self.money == 0:
            self.check()
            return 0, self.max_bet

        old_round_bet = self.round_bet

        num_cards_shown = len(self.table_cards_so_far)

        # raise encodings for when not all ([0, k - 2]) other players have reacted to the latest information?

        k = self.num_players - 1 # same as defined in Line 238.
        rbs, tot_rbs, money_amts, folds, board_encodings = [0] * k, [0] * k, [1] * k, [False] * k, [[-1] * 10] * k
        if len(self.round_action_dict["other_players_round_bets"]) > 0:
            rbs, tot_rbs, money_amts, folds, board_encodings = self.round_action_dict["other_players_round_bets"][-1], self.round_action_dict["other_players_total_round_bets"][-1], self.round_action_dict["other_players_money"][-1], self.round_action_dict["has_folded"][-1], self.round_action_dict["other_players_board_card_encodings"][-1]
        
        other_player_info = []
        k = len(rbs)

        for i in range(k):
            other_player_info.extend(board_encodings[i])
            if folds[i]:
                other_player_info.append(-1)
#            elif rbs[i] == money_amts[i]:
#                other_player_info.append(1)
            else:
                other_player_info.append(rbs[i] / self.buy_in)
            other_player_info.append(tot_rbs[i] / self.buy_in)
        
        encoded_hand_cards = self._encode_cards(self._hand, False)
        encoded_board_cards = self._encode_cards(self.table_cards_so_far)
        state_vec = torch.cat((torch.tensor(encoded_hand_cards, device = device).unsqueeze(0), torch.tensor(encoded_board_cards, device = device).unsqueeze(0), torch.tensor([self.money / self.buy_in], device = device).unsqueeze(0), torch.tensor(other_player_info, device = device).unsqueeze(0)), dim = 1).to(dtype = torch.float32) 

        #debug_print(f"state_vec: {state_vec}")
        self.nn_vals["state_vecs"].append(state_vec)

        # The policy network now outputs probabilities for fold, call, and raise,
        # and a parameter for the raise amount distribution.
        if not self.train_network:
            with torch.no_grad():
                action_probs, db_param = self.policy_nn(state_vec)
        else:
            action_probs, db_param = self.policy_nn(state_vec)

        #debug_print(f"db_param: {db_param}")

        # Create a categorical distribution to sample the action
        action_dist = torch.distributions.Categorical(action_probs)
        action_index = action_dist.sample()  # 0: fold, 1: call/check, 2: raise

        # Store log probability of the chosen action for training
        log_prob_action = action_dist.log_prob(action_index)

        # The old `win_probs` is replaced by `action_probs`.
        # Note: This will likely require changes to the training loop (train_iter).
        self.nn_vals["action_probs"].append(action_probs)

        if self.money == 0 or self.round_bet == self.max_bet:
            self.nn_vals["expected_fold_prob_zero"].append(True)
        else:
            self.nn_vals["expected_fold_prob_zero"].append(False)
        
        # Execute the sampled action
        if action_index == 0:  # Fold
            self.fold()
        elif action_index == 1 and (self.money == 0 or self.round_bet == self.max_bet):  # Call/Check
            self.check()
        elif action_index == 1:
            try:
                self.call()
            except ValueError:
                self.max_bet = self.all_in()
        else:  # Raise
            # n is the number of min_bet chunks the player can afford to raise by, after calling
            money_after_call = self.money - (self.max_bet - self.round_bet)
            n = max(0, money_after_call // self.min_bet - 1)

            if n > 0:
                self.is_latest_action_raise = True
                self.max_bet_before_raise = self.max_bet

                # Sample x from Binomial(n, p)
                binomial_dist = torch.distributions.Binomial(n, db_param.squeeze())
                x = torch.tensor([binomial_dist.sample()], dtype = torch.float32, device = device)

                # Manually calculate log probability of x for gradient purposes
                p = db_param.squeeze(-1)
                n_t = torch.tensor([n], dtype = torch.float32, device = device)
                log_binom_coeff = (torch.lgamma(n_t + 1) -
                                   torch.lgamma(x + 1) -
                                   torch.lgamma(n_t - x + 1))
                log_prob_x = log_binom_coeff + x * torch.log(p) + (n_t - x) * torch.log(1 - p)

                #debug_print(f"binomial_sample_log_prob: {log_prob_x}")
                log_prob_action += log_prob_x
                
                # The amount to raise is on top of the call amount
                raise_amount = (1 + x.item()) * self.min_bet
                self.max_bet = self.raise_bet(raise_amount)
            elif money_after_call == self.min_bet:
                self.is_latest_action_raise = True
                self.max_bet_before_raise = self.max_bet
                self.max_bet = self.raise_bet(self.min_bet)
            elif self.money == 0:
                self.check()
            else:
                # Cannot raise if n <= 0, so just call
                try:
                    self.call()
                except ValueError:
                    self.max_bet = self.all_in()

        #debug_print(f"log_prob_action: {log_prob_action}")
        self.nn_vals["log_prob_action"].append(log_prob_action)

        self.round_action_dict["self_round_bets"].append(self.round_bet - old_round_bet)
        self.round_action_dict["pots"].append(self.pot + self.round_bet - old_round_bet)
        self.round_action_dict["raise_rewards"].append(0)

        if not self.train_network:
            with torch.no_grad():
                V_t = self.critic_nn(state_vec)
        else:
            V_t = self.critic_nn(state_vec)
        self.nn_vals["critic_outputs"].append(V_t)

        add_to_pot = self.round_bet - old_round_bet

#        end_time = time.time()
#        elapsed_ms = (end_time - start_time) * 1000  # convert to milliseconds
        # print(f"cue_for_action took {elapsed_ms:.2f} ms to run.")

        return add_to_pot, self.max_bet

    def collect_trajectory(self):
        if len(self.round_action_dict["neg_action_regrets"]) == 0:
            return

        # Store the data for this hand as a single trajectory
        trajectory = {
            "rewards": self.round_action_dict["neg_action_regrets"],
            "log_probs": torch.cat(self.nn_vals["log_prob_action"], dim = 0),
            "critic_values": torch.cat(self.nn_vals["critic_outputs"], dim = 0),
            "action_probs": torch.cat(self.nn_vals["action_probs"], dim = 0),
            "fold_penalty_mask": torch.tensor(self.nn_vals["expected_fold_prob_zero"], device = device)
        }
        self.trajectory_buffer.append(trajectory)

    def train_batch(self, is_terminal = False):
        if len(self.trajectory_buffer) == 0:
            return

        # Unpack trajectories into batches
        batch_rewards = []
        batch_log_probs = []
        batch_critic_values = []
        batch_action_probs = []
        batch_fold_penalty_masks = []

        for trajectory in self.trajectory_buffer:
            batch_rewards.extend(trajectory["rewards"])
            batch_log_probs.append(trajectory["log_probs"])
            batch_critic_values.append(trajectory["critic_values"])
            batch_action_probs.append(trajectory["action_probs"])
            batch_fold_penalty_masks.append(trajectory["fold_penalty_mask"])

        rewards = torch.tensor(batch_rewards).to(dtype = torch.float32, device = device).unsqueeze(1)
        log_probs = torch.cat(batch_log_probs, dim = 0).to(dtype = torch.float32, device = device).unsqueeze(1)
        critic_values = torch.cat(batch_critic_values, dim = 0).to(dtype = torch.float32, device = device)
        action_probs = torch.cat(batch_action_probs, dim = 0).to(dtype = torch.float32, device = device)
        fold_penalty_mask = torch.cat(batch_fold_penalty_masks, dim = 0).to(device = device)
        
        if is_terminal:
            # Calculate advantages for terminal state
            with torch.no_grad():
                critic_values_next = torch.cat([critic_values_next, tensor([[0.0]], device = device)], dim = 0).detach()
                advantages = rewards + self.gamma * critic_values_next - critic_values

            # Policy loss includes the last action
            policy_loss = -(log_probs * advantages).mean()

            critic_loss = F.mse_loss(rewards + self.gamma * critic_values_next, critic_values)

        else:
            # Calculate advantages
            with torch.no_grad():
                critic_values_next = critic_values[1 : ].detach()
                advantages = rewards[ : -1] + self.gamma * critic_values_next - critic_values[ : -1]

            # Policy loss
            policy_loss = -(log_probs[ : -1] * advantages).mean()
            
            critic_loss = F.mse_loss(rewards[ : -1] + self.gamma * critic_values_next, critic_values[ : -1])

        # Fold loss
        # Penalize the network for assigning a non-zero probability to folding when it
        # should be forced to check/call (e.g., when all-in or max_bet is met).
        # The fold_penalty_mask is True for these specific states.
        fold_probs_to_penalize = action_probs[fold_penalty_mask, 0]  # Select fold prob (col 0) for relevant rows
        if fold_probs_to_penalize.numel() > 0:
            # Explicitly calculate L2 loss (MSE) against a target of zero.
            fold_loss = F.mse_loss(fold_probs_to_penalize, torch.zeros_like(fold_probs_to_penalize))
        else:
            fold_loss = torch.tensor(0.0, device = device)

        # Total policy loss
        total_policy_loss = policy_loss + self.lmbda * fold_loss

        self.plot_data_game["critic_losses"].append(critic_loss.item())
        self.plot_data_game["policy_losses"].append(policy_loss.item())
        self.plot_data_game["fold_losses"].append(fold_loss.item())
        self.plot_data_game["total_policy_losses"].append(total_policy_loss.item())
        self.plot_data_game["rewards"].append(torch.sum(rewards))
        self.plot_data_game["money_values"].append(self.money)
        self.plot_data_game["batch_sizes"].append(len(batch_rewards))

        #debug_print(f"rewards: {rewards}")
        #debug_print(f"log_probs: {log_probs}")
        #debug_print(f"critic_values: {critic_values}")
        #debug_print(f"action_probs: {action_probs}")
        #debug_print(f"advantages: {advantages}")
        #debug_print(f"fold_penalty_mask: {fold_penalty_mask}")
        #debug_print(f"critic_loss: {critic_loss}")
        #debug_print(f"policy_loss: {policy_loss}")
        #debug_print(f"fold_loss: {fold_loss}")

        # Backpropagation
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_nn.parameters(), max_norm = 0.1)
        self.policy_optimizer.step()
        self.policy_scheduler.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_nn.parameters(), max_norm = 0.1)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        self.plot_data_game["policy_grad_norms"].append(policy_grad_norm.item())
        self.plot_data_game["critic_grad_norms"].append(critic_grad_norm.item())

        self.num_updates += 1
        if self.num_updates % 50 == 0:
            self.save_model()
        
        # Clear the buffer after training
        self.trajectory_buffer.clear()

    def save_model(self):
        if not self.train_network:
            return
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        policy_checkpoint_path = f"checkpoints/policy/policy_checkpoint_{self.name}_{now}_{self.num_updates}_{self.original_num_players}.pth"
        critic_checkpoint_path = f"checkpoints/critic/critic_checkpoint_{self.name}_{now}_{self.num_updates}_{self.original_num_players}.pth"
        os.makedirs(os.path.dirname(policy_checkpoint_path), exist_ok = True)
        os.makedirs(os.path.dirname(critic_checkpoint_path), exist_ok = True)
        torch.save(self.policy_nn.state_dict(), policy_checkpoint_path)
        torch.save(self.critic_nn.state_dict(), critic_checkpoint_path)
        return policy_checkpoint_path, critic_checkpoint_path, self.policy_optimizer.param_groups[0]['lr'], self.num_updates
