from treys import Card, Evaluator, Deck
from base_classes.player import Player
from base_classes.plotter import Plotter
from main.automated_player import AutomatedPlayer

class Game:
    # Constructor (initializes attributes)
    def __init__(self, num_players, buy_in, min_bet, one_bot = False, all_bots = False, load_checkpt_policy = "", load_checkpt_critic = "", train_network = False, lmbda = 0.1, plotter = None):
        self.plotter = plotter
        self.suits = ['h', 'd', 'c', 's']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.deck_string = [f'{rank}{suit}' for suit in self.suits for rank in self.ranks]
        self.deck = Deck()
        self.num_players = num_players
        self.num_players_threshold = 10
        self.max_bet = 0
        self.players = []
        self.one_bot = one_bot
        self.all_bots = all_bots
        self.min_bet = min_bet
        if self.one_bot:
            self.players.append(AutomatedPlayer("P0", buy_in, self.num_players, min_bet, load_checkpt_policy, load_checkpt_critic, train_network, lmbda))
            for i in range(1, num_players):
                self.players.append(Player(f"P{i}", buy_in, self.num_players, min_bet))
        elif self.all_bots:
            for i in range(num_players):
                self.players.append(AutomatedPlayer(f"P{i}", buy_in, self.num_players, min_bet, load_checkpt_policy[i], load_checkpt_critic[i], train_network, lmbda))
        else:
            for i in range(num_players):
                self.players.append(Player(f"P{i}", buy_in, self.num_players, min_bet))
        self.table_cards_string = []
        self.table_cards = []
        self.pretty_table_cards = []
        self.cards_shown = 0
        self.num_active = self.num_players
        self.active_indices = set(range(self.num_players))
        self.pot = 0
        self.lost = 0
        self.evaluator = Evaluator()
        self.player_cards = []
        self.removed_automated_players = []

    def reset(self):
        self.max_bet = 0
        self.pot = 0
        self.deck = Deck()
        self.table_cards_string = []
        self.table_cards = []
        self.pretty_table_cards = []
        self.cards_shown = 0
        self.lost += len(self.players) - self.num_players
        self.num_players = len(self.players)
        self.num_active = self.num_players
        self.active_indices = set(range(self.num_players))
        self.player_cards = []

    def distribute_cards(self):
        if self.num_players > 10:
            raise ValueError(f"Number of players {self.num_players} exceeds the threshold {self.threshold}. Exiting program.")
        self.player_cards = self.deck.draw(2 * self.num_players + 5)
        self.player_cards = [Card.int_to_str(card) for card in self.player_cards]
        for i in range(self.num_players):
            self.players[i].set_hand(self.player_cards[2 * i : 2 * i + 2])
        self.table_cards_string = self.player_cards[2 * self.num_players : 2 * self.num_players + 5]
        self.table_cards = [Card.new(table_card_string) for table_card_string in self.table_cards_string]
        self.pretty_table_cards = [Card.int_to_pretty_str(table_card) for table_card in self.table_cards]

    def betting_round(self, is_first = False):
        print("-------Betting Round-------")
        i = 0
        if is_first and self.num_players > 2:
            i = 2
        players_since_no_raise = 0
        updated_table_cards = [False for j in range(self.num_players)]
        while players_since_no_raise < self.num_players:
            if not updated_table_cards[i] and not is_first and isinstance(self.players[i], AutomatedPlayer):
                self.players[i].update_table_cards(self.table_cards_string[ : self.cards_shown])
                updated_table_cards[i] = True
            if self.players[i].is_active():
                old_max_bet = self.max_bet
                old_round_bet, old_money = self.players[i].round_bet, self.players[i].money
                add_to_pot, self.max_bet = self.players[i].cue_for_action()
                self.pot += add_to_pot
                if self.max_bet != old_max_bet:
                    players_since_no_raise = 1
                else:
                    players_since_no_raise += 1
                for j in range(self.num_players):
                    self.players[j].set_max_bet(self.max_bet, self.pot)
                    if isinstance(self.players[j], AutomatedPlayer):
                        if not self.players[i].is_active():
                            self.players[j].decrement_num_players_round()
                        if i != j:
                            if self.players[i].is_active() and self.players[j].is_latest_action_raise:
                                x = min(old_money, self.players[j].round_bet - old_round_bet) - min(old_money, self.players[j].max_bet_before_raise - old_round_bet)
                                self.players[j].update_action_dict(add_to_pot, self.players[i].money, not self.players[i].is_active(), self.cards_shown, x)
                            else:
                                self.players[j].update_action_dict(add_to_pot, self.players[i].money, not self.players[i].is_active(), self.cards_shown)
                if not self.players[i].is_active():
                    self.num_active -= 1
                    self.active_indices.remove(i)
                if self.num_active == 1:
                    break
            else:
                players_since_no_raise += 1
                for j in range(self.num_players):
                    if isinstance(self.players[j], AutomatedPlayer) and i != j:
                        self.players[j].update_action_dict(0, self.players[i].money, True, self.cards_shown)
            i += 1
            i = i % self.num_players
        print(f"Current Pot: {self.pot}")
        
    def show_table_cards(self, stage):
        print("-------Showing Cards-------")
        if stage == 0:
            self.cards_shown = 3
            print(" | ".join(self.pretty_table_cards[ : 3]), "? | ?", sep = " | ")
        elif stage == 1:
            self.cards_shown = 4
            print(" | ".join(self.pretty_table_cards[ : 4]), "?", sep = " | ")
        elif stage == 2:
            self.cards_shown = 5
            print(" | ".join(self.pretty_table_cards))

    def round_end(self, winner_indices, winner_score):
        print("-------Player Money-------")
        ctr = 0
        small_blind_in = True
        for i in range(self.num_players):
            print(f"{self.players[ctr].get_name()}: {self.players[ctr].get_money()}", end = " | ")
            if self.players[ctr].get_money() == 0:
                if isinstance(self.players[ctr], AutomatedPlayer):
                    self.removed_automated_players.append(self.players[ctr])
                self.players.pop(ctr)
                if ctr == 0:
                    small_blind_in = False
            else:
                ctr += 1
        ind = 2 if small_blind_in else 1
        while len(self.players) > 1 and self.players[ind % len(self.players)].get_money() < 2 * self.min_bet:
            if isinstance(self.players[ind % len(self.players)], AutomatedPlayer):
                self.removed_automated_players.append(self.players[ind % len(self.players)])
            self.players.pop(ind % len(self.players))
        self.reset()
        for i in range(self.num_players):
            self.players[i].update_num_players(self.num_players)
            self.players[i].reset()
        print()
        if small_blind_in:
            self.players = self.players[1 : ] + self.players[ : 1]

    def strongest_hand_indices(self, player_hands, active_indices):
        # Convert strings to treys Card objects
        board = self.table_cards

        scores = []
        for hand in player_hands:
            hole_cards = [Card.new(c) for c in hand]
            score = self.evaluator.evaluate(board, hole_cards)
            scores.append(score)

        # Find the best (minimum) score
        best_score = min(scores)

        # Find all indices with the best score (to handle ties)
        best_indices = [active_indices[i] for i, s in enumerate(scores) if s == best_score]
        
        return best_score, best_indices

    def round(self):
        add_to_pot, self.max_bet = self.players[0].small_blind()
        self.pot += add_to_pot
        for i in range(self.num_players):
            self.players[i].set_max_bet(self.max_bet, self.pot)
        add_to_pot, self.max_bet = self.players[1].big_blind()
        self.pot += add_to_pot
        for i in range(self.num_players):
            self.players[i].set_max_bet(self.max_bet, self.pot)          
        self.distribute_cards()
        self.betting_round(True)
        ended_early = False
        if self.num_active > 1:
            self.show_table_cards(0)
            self.betting_round()
        else:
            ended_early = True
        if self.num_active > 1:
            self.show_table_cards(1)
            self.betting_round()
        else:
            ended_early = True
        if self.num_active > 1:
            self.show_table_cards(2)
            self.betting_round()
        else:
            ended_early = True
        player_hands = []
        active_indices = []
        if ended_early:
            self.show_table_cards(2)
        print("-------Player Hands-------")
        for index in range(self.num_players):
            if index in self.active_indices:
                player_hands.append(self.player_cards[2 * index : 2 * index + 2])
                active_indices.append(index)
                print(f"{self.players[index].get_name()}: ", end = "")
            else:
                print(f"{self.players[index].get_name()} (Folded): ", end = "")
            self.players[index].show_hand(" | ")
        print()
        winner_score, winner_indices = self.strongest_hand_indices(player_hands, active_indices)
        num_winners = len(winner_indices)
        for i in range(self.num_players):
            if isinstance(self.players[i], AutomatedPlayer) and self.players[i].train_network:
                self.players[i].compute_action_regrets(len(winner_indices), i in winner_indices, winner_score, self.table_cards_string)
                self.players[i].train_iter()
        for winner_index in winner_indices:
            self.players[winner_index].add_to_money(self.pot / num_winners)
        winner_names = [self.players[i].get_name() for i in winner_indices]
        self.round_end(winner_indices, winner_score)
        if num_winners == 1:
            print(", ".join(x for x in winner_names), "wins the pot.", sep = " ")
        else:
            print(", ".join(x for x in winner_names), "split the pot.", sep = " ")

    def end(self):
        player_info = []
        all_automated_players = self.removed_automated_players + [p for p in self.players if isinstance(p, AutomatedPlayer)]
        
        for player in all_automated_players:
            policy_checkpoint_path, critic_checkpoint_path = player.save_model()
            player_info.append((player.get_name(), policy_checkpoint_path, critic_checkpoint_path))

        # Sort by player name number to handle P1, P2, ... P10 correctly
        player_info.sort(key = lambda x: int(x[0][1 : ]))

        policy_checkpoint_paths = [info[1] for info in player_info]
        critic_checkpoint_paths = [info[2] for info in player_info]

        if self.plotter:
            plot_data = [player.plot_data_game for player in all_automated_players]
            self.plotter.log_data(plot_data)

        return policy_checkpoint_paths, critic_checkpoint_paths