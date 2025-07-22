from treys import Card

class Player:
    def __init__(self, name, buy_in, num_players, min_bet):
        self.name = name
        self.money = buy_in
        self.num_players = num_players
        self.active = True
        self._hand = []
        self.round_bet = 0
        self.max_bet = 0
        self.min_bet = min_bet
        self.buy_in = buy_in
        self.is_small_blind = False
        self.is_big_blind = False

    def update_num_players(self, new_val):
        self.num_players = new_val

    def reset(self):
        self._hand = [] 
        self.active = True
        self.round_bet = 0
        self.max_bet = 0
        self.is_small_blind = False
        self.is_big_blind = False

    def set_hand(self, cards):
        self._hand = cards
    
    def show_hand(self, end_with = "\n"):
        print(Card.int_to_pretty_str(Card.new(self._hand[0])), Card.int_to_pretty_str(Card.new(self._hand[1])), sep = " ", end = end_with)

    def is_active(self):
        return self.active

    def get_name(self):
        return self.name

    def get_money(self):
        return self.money

    def set_max_bet(self, max_bet, pot):
        self.max_bet = max_bet

    def add_to_money(self, pot):
        self.money += pot

    def small_blind(self):
        self.money -= self.min_bet
        self.round_bet += self.min_bet
        self.max_bet = self.min_bet
        self.is_small_blind = True
        return self.min_bet, self.max_bet

    def big_blind(self):
        self.money -= 2 * self.min_bet
        self.round_bet += 2 * self.min_bet
        self.max_bet = 2 * self.min_bet
        self.is_big_blind = True
        return 2 * self.min_bet, self.max_bet

    def check(self):
        if self.money != 0 and self.max_bet > self.round_bet:
            raise ValueError(f"Player {self.name} cannot check since {self.round_bet} (player's bet) < {self.max_bet} (current round bet). Exiting program.")
        print(f"Player {self.name}: Check")

    def call(self):
        if self.max_bet - self.round_bet > self.money:
            raise ValueError(f"Player {self.name} doesn't have money to call. Exiting program.")
        additional_bet = self.max_bet - self.round_bet
        self.money -= additional_bet
        self.round_bet = self.max_bet
        print(f"Player {self.name}: Call")

    def raise_bet(self, raise_by):
        if self.max_bet + raise_by - self.round_bet > self.money:
            raise ValueError(f"Player {self.name} doesn't have money to raise by {raise_by}. Exiting program.")
        additional_bet = self.max_bet + raise_by - self.round_bet
        self.money -= additional_bet
        self.round_bet = self.max_bet + raise_by
        print(f"Player {self.name}: Raise by {raise_by}")
        return self.round_bet
    
    def all_in(self):
        if self.money == 0:
            raise ValueError(f"Player {self.name} is already all in. Exiting program.")
        self.round_bet += self.money 
        self.money = 0
        print(f"Player {self.name}: All In")
        return max([self.max_bet, self.round_bet])

    def fold(self):
        self.active = False
        print(f"Player {self.name}: Fold")

    def is_integer(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def cue_for_action(self):
        self.show_hand()
        while True:
            try:
                action = input(f"Enter action for Player {self.name} - ").split()
                old_round_bet = self.round_bet
                if len(action) == 1:
                    if action[0] == "Check":
                        self.check()
                    elif action[0] == "Call":
                        self.call()
                    elif action[0] == "AllIn":
                        self.max_bet = self.all_in()
                    elif action[0] == "Fold":
                        self.fold()
                    else:
                        raise ValueError(f"Unrecognised action {action[0]}.")
                elif len(action) == 2:
                    if action[0] == "Raise" and self.is_integer(action[1]):
                        self.max_bet = self.raise_bet(int(action[1]))
                    else:
                        raise ValueError(f"Unrecognised action.")
                else:
                    raise ValueError(f"Unrecognised action.")
                break
            except ValueError:
                print("Invalid input, re-enter action.")
        add_to_pot = self.round_bet - old_round_bet
        return add_to_pot, self.max_bet        