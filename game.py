from player import Player
from automated_player import AutomatedPlayer
from treys import Card, Evaluator, Deck

class Game:
    # Constructor (initializes attributes)
    def __init__(self, num_players, buy_in, min_bet, play_with_bot = False, load_checkpt_policy = "", load_checkpt_critic = "", train_network = False, lmbda = 0.1):
        self.suits = ['h', 'd', 'c', 's']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.deck_string = [f'{rank}{suit}' for suit in self.suits for rank in self.ranks]
        self.deck = Deck()
        self.num_players = num_players
        self.num_players_threshold = 10
        self.max_bet = 0
        self.players = []
        self.play_with_bot = play_with_bot
        if self.play_with_bot:
            self.players.append(AutomatedPlayer("P0", buy_in, self.num_players, min_bet, load_checkpt_policy, load_checkpt_critic, train_network, lmbda))
            for i in range(1, num_players):
                self.players.append(Player(f"P{i}", buy_in, self.num_players, min_bet))
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
        while players_since_no_raise < self.num_players:
            if self.players[i].is_active():
                old_max_bet = self.max_bet
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
                            self.players[j].update_action_dict(add_to_pot, self.players[i].money)
                if not self.players[i].is_active():
                    self.num_active -= 1
                    self.active_indices.remove(i)
                if self.num_active == 1:
                    break
            else:
                players_since_no_raise += 1
                for j in range(self.num_players):
                    if isinstance(self.players[j], AutomatedPlayer):
                        self.players[j].update_action_dict(0, self.players[i].money)
                        break
            i += 1
            i = i % self.num_players
        print(f"Current Pot: {self.pot}")
        
    def show_table_cards(self, stage):
        print("-------Showing Cards-------")
        if stage == 0:
            print(" | ".join(self.pretty_table_cards[ : 3]), "? | ?", sep = " | ")
        elif stage == 1:
            print(" | ".join(self.pretty_table_cards[ : 4]), "?", sep = " | ")
        elif stage == 2:
            print(" | ".join(self.pretty_table_cards))

    def round_end(self, winner_indices, winner_score):
        print("-------Player Money-------")
        ctr = 0
        for i in range(self.num_players):
            if isinstance(self.players[ctr], AutomatedPlayer) and self.players[ctr].train_network:
                self.players[ctr].compute_action_regrets(len(winner_indices), i in winner_indices, winner_score, self.table_cards_string)
                self.players[ctr].train_iter()
            print(f"{self.players[ctr].get_name()}: {self.players[ctr].get_money()}", end = " | ")
            if self.players[ctr].get_money() == 0:
                self.players.pop(ctr)
            else:
                ctr += 1
        self.reset()
        for i in range(self.num_players):
            self.players[i].update_num_players(self.num_players)
            self.players[i].reset()
        print()

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
        if self.num_players > 2:
            add_to_pot, self.max_bet = self.players[1].big_blind()
        else:
            add_to_pot, self.max_bet = self.players[1].small_blind()
        self.pot += add_to_pot
        for i in range(self.num_players):
            self.players[i].set_max_bet(self.max_bet, self.pot)          
        self.distribute_cards()
        self.betting_round(True)
        if self.num_active > 1:
            self.show_table_cards(0)
            self.betting_round()
        if self.num_active > 1:
            self.show_table_cards(1)
            self.betting_round()
        if self.num_active > 1:
            self.show_table_cards(2)
            self.betting_round()
        player_hands = []
        active_indices = []
        print("-------Player Hands-------")
        for active_player_index in self.active_indices:
            player_hands.append(self.player_cards[2 * active_player_index : 2 * active_player_index + 2])
            active_indices.append(active_player_index)
            print(f"{self.players[active_player_index].get_name()}: ", end = "")
            self.players[active_player_index].show_hand(" | ")
        print()
        winner_score, winner_indices = self.strongest_hand_indices(player_hands, active_indices)
        num_winners = len(winner_indices)
        for winner_index in winner_indices:
            self.players[winner_index].add_to_money(self.pot / num_winners)
        winner_names = [self.players[i].get_name() for i in winner_indices]
        self.round_end(winner_indices, winner_score)
        if num_winners == 1:
            print(", ".join(x for x in winner_names), "wins the pot.", sep = " ")
        else:
            print(", ".join(x for x in winner_names), "split the pot.", sep = " ")
        if self.num_players > 2:
            self.players = self.players[1 : ] + self.players[ : 1]    