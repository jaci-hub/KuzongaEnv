'''
Name: Jacinto Jeje Matamba Quimua
Date: 10/28/2025

This is the python gym-style API for my game Divide21
'''

import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils.env_checker import check_env
import warnings


class Divide21Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, digits=2, players=None, render_mode=None, auto_render=False):
        super().__init__()
        self.players = players or []
        self.dynamic_number = None
        self.player_turn = 0
        self.digits = digits
        self.available_digits_per_index = {i: list(range(10)) for i in range(digits)}
        self.maxScore = 9*digits
        self.render_mode = render_mode
        
        warnings.filterwarnings(
            "ignore",
            message=".*Box observation space maximum and minimum values are equal.*"
        )

        # (1) Action space:
        # action is dictionary with keys: division, digit, index
        #   division (bool): true/false
        #   digit (int): if division=true, then it is the divisor, else it is the new digit in the index chosen
        #   index (int): if division=true, then it is None, else the index where the digit will be overwriten
        self.action_space = spaces.Dict({
            "division": spaces.Discrete(2),
            "digit": spaces.Discrete(10),
            "index": spaces.Discrete(digits)
        })

        # (2) Observation space: 
        # observation is a dictionary with keys: dynamic_number, available_digits_per_index, players, player_turn]
        #   dynamic_number (str): the current value of the number whose digits are manipulated
        #   available_digits_per_index (dict): a dictionary where the keys are the indexes of the dynamic_number and their values are the list of digits available at that index
        #   players (list): the list of dictionaries with each player's id, score and a variable (is_current_turn) that tells if it is the player's turn to play
        #   player_turn (int): the id of the player with the turn
        number_of_players = max(1, len(self.players)) # since spaces.Discrete(n) goes from 0 to n-1, so if players=[], then number_of_players will still be 1 so spaces.Discrete(1) will just be 0
        self.observation_space = spaces.Dict({
            "dynamic_number": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "available_digits_per_index": spaces.MultiBinary(10 * digits),
            "players": spaces.Box(
                low=np.array([0, -self.maxScore, 0] * number_of_players, dtype=np.int64),
                high=np.array([number_of_players - 1, self.maxScore, 1] * number_of_players, dtype=np.int64),
                shape=(number_of_players * 3,),
                dtype=np.int64
            ),
            "player_turn": spaces.Discrete(number_of_players)
        })
    
    
    def _encode_players(self):
        '''
        Encodes player info numerically:
        Each player has attributes: id, score, is_current_turn
            Note: if is_current_turn=1, it is the player's turn to play, else, it is not 
        '''
        if not self.players:
            # create a default single-player representation
            encoded = np.zeros((1, 3), dtype=np.int64)
            encoded[0] = [0, 0, 1]  # id=0, score=0, is_current_turn=1
            return encoded.flatten()
        
        num_players = len(self.players)
        encoded = np.zeros((num_players, 3), dtype=np.int64)
        for i, p in enumerate(self.players):
            encoded[i, 0] = p.get("id", i)
            encoded[i, 1] = p.get("score", 0)
            encoded[i, 2] = 1 if i == self.player_turn else 0
        return encoded.flatten()
    
    
    def _create_dynamic_number(self, max_attempts=10_000):
        '''
        generate a valid starting number
        '''
        rng = self.np_random  # gym seeding
        for _ in range(max_attempts):
            # sample digits 0-9
            digits_arr = rng.integers(0, 10, size=self.digits)
            # ensure first digit != 0
            if digits_arr[0] == 0:
                digits_arr[0] = rng.integers(1, 10)
            s = "".join(str(int(d)) for d in digits_arr)  # string form
            # check divisibility condition using Python int (arbitrary precision)
            try:
                num = int(s)
            except Exception:
                continue
            if math.gcd(num, 210) == 1:
                return s

        # fallback: deterministic string like "10...01"
        fallback = "1" + ("0" * (self.digits - 2)) + "1"
        return fallback
    
    def _get_prohibited_digit_list_at_index(self, index):
        prohibited = set()
        # no leading zero
        if index == 0:
            prohibited.add(0)

        # can’t make number 0 or 1
        for d in [0, 1]:
            modified = self.dynamic_number[:index] + str(d) + self.dynamic_number[index + 1:]
            if int(modified) in (0, 1):
                prohibited.add(d)

        return list(prohibited)
    
    def _remove_all_prohibited_digits_at_given_index_from_given_list(self, index, digit_list):
        prohibited_digits = set(self._get_prohibited_digit_list_at_index(index))
        return [d for d in digit_list if d not in prohibited_digits]
    
    def _setup_available_digits_per_index(self):
        available_digits_per_index = {}

        for i in range(self.digits):
            current_digit = int(self.dynamic_number[i])
            all_digits = [d for d in range(10) if d != current_digit]
            filtered_digits = self._remove_all_prohibited_digits_at_given_index_from_given_list(i, all_digits)
            available_digits_per_index[i] = filtered_digits

        return available_digits_per_index
    
    def _encode_available_digits(self):
        mask = np.zeros((self.digits, 10), dtype=np.int64)
        for idx, available in self.available_digits_per_index.items():
            mask[idx, available] = 1
        return mask.flatten()

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.dynamic_number = self._create_dynamic_number()
        self.available_digits_per_index = self._setup_available_digits_per_index()
        self.player_turn = 0
        obs = {
            "dynamic_number": np.array([int(d) for d in self.dynamic_number], dtype=np.int8),
            "available_digits_per_index": self._encode_available_digits(),
            "players": self._encode_players(),
            "player_turn": np.int64(self.player_turn)
        }
        info = {"seed": seed}
        return obs, info
    
    
    def _index_available_digit_list_is_empty(self, index):
        if not self.available_digits_per_index:
            return True

        index_available_digit_list = self.available_digits_per_index[index]
        return len(index_available_digit_list) == 0
    
    def _update_available_digits_per_index(self, index=None):
        if index is None: # division was performed
            for i in range(self.digits):
                current_digit = int(self.dynamic_number[i])
                index_available_digit_list = [d for d in self.available_digits_per_index[i] if d != current_digit]
                index_available_digit_list = self._remove_all_prohibited_digits_at_given_index_from_given_list(i, index_available_digit_list)

                if index_available_digit_list:
                    self.available_digits_per_index[i] = index_available_digit_list
                else:
                    all_digits = [d for d in range(10) if d != current_digit]
                    all_digits = self._remove_all_prohibited_digits_at_given_index_from_given_list(i, all_digits)
                    self.available_digits_per_index[i] = all_digits
        else: # division was not performed
            if not self._index_available_digit_list_is_empty(index):
                return

            current_digit = int(self.dynamic_number[index])
            all_digits = [d for d in range(10) if d != current_digit]
            all_digits = self._remove_all_prohibited_digits_at_given_index_from_given_list(index, all_digits)
            self.available_digits_per_index[index] = all_digits
    
    def _remove_each_quotient_digit_from_available_digits_per_index(self, quotient_string):
        if not self.available_digits_per_index:
            return

        for i in range(len(quotient_string)):
            index_available_digit_list = self.available_digits_per_index[i]
            if not index_available_digit_list:
                continue

            digit_to_remove = int(quotient_string[i])
            index_available_digit_list = [d for d in index_available_digit_list if d != digit_to_remove]
            self.available_digits_per_index[i] = index_available_digit_list
    
    def _remove_digit_from_index_available_digits(self, index, digit_to_remove):
        if not self.available_digits_per_index:
            return

        index_available_digit_list = self.available_digits_per_index[index]
        if not index_available_digit_list:
            return

        index_available_digit_list = [d for d in index_available_digit_list if d != digit_to_remove]
        self.available_digits_per_index[index] = index_available_digit_list
    
    def _game_over(self):
        # (1) quotient 1
        if int(self.dynamic_number) == 1:
            return True
        # (2) max points
        for player in self.players:
            if player['score'] >= self.maxScore:
                return True
        # (3) only one player left without -max points or less
        count = 0
        for player in self.players:
            if player['score'] <= -self.maxScore:
                count += 1
        if count == len(self.players) - 1:
            return True
        
        return False
    
    def step(self, action):
        """
        Executes one step of the Divide21 environment.
        Args:
            action (dict): {
                "division": 0 or 1,
                "digit": int,
                "index": int (ignored if division == 1)
            }
        Returns:
            obs, reward, terminated, truncated, info
        """
        division = bool(action["division"])
        digit = int(action["digit"])
        index = int(action["index"])
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # (1) Division attempt
        if division:
            num = int(self.dynamic_number)
            if digit in [0, 1]: # not allowed to divide by 0 or 1
                reward = -2*self.maxScore
                # update player score
                if self.players:
                    self.players[self.player_turn]["score"] -= 2*self.maxScore
                info["message"] = "Division by 0 or 1 is not allowed!"
            elif num % digit == 0:
                new_num = num // digit
                new_num = str(new_num)
                # keep the original number of digits
                if len(new_num) < self.digits:
                    new_num = "0"*(self.digits - len(new_num)) + new_num
                self.dynamic_number = new_num
                reward = 1
                # update the list of available digits per index
                #   (1) remove each quotient digit from available digits per index
                self._remove_each_quotient_digit_from_available_digits_per_index(self.dynamic_number)
                #   (2) update available digits per index
                self._update_available_digits_per_index(index)
                # update player score
                if self.players:
                    self.players[self.player_turn]["score"] += digit
                info["message"] = f"Divided by {digit}"
            else:
                reward = -1
                # update player score
                if self.players:
                    self.players[self.player_turn]["score"] -= digit
                info["message"] = f"Careful, {digit} is not a factor of {int(self.dynamic_number)}"
        # (2) Digit change
        else:
            if digit in self.available_digits_per_index[index]:
                num_str = list(self.dynamic_number)
                num_str[index] = str(digit)
                self.dynamic_number = "".join(num_str)
                reward = 1
                # update the list of available digits per index
                # (1) remove digit from index available digits
                self._remove_digit_from_index_available_digits(index, digit)
                # (2) update available digits per index
                self._update_available_digits_per_index(index)
                # update player turn
                if self.players:
                    self.player_turn = (self.player_turn + 1) % len(self.players)
                    self.players[self.player_turn]['is_current_turn'] = 1
                    for player in self.players:
                        if player['id'] != self.players[self.player_turn]['id']:
                            player['is_current_turn'] = 0
                info["message"] = f"Updated digit at index {index} to {digit}"
            else:
                reward = -1
                info["message"] = f"Cannot update the digit at index {index} to {digit}"

        # Check if game is over
        if self._game_over():
            terminated = True
            
            if reward > 0:
                reward += 1
            else:
                reward -= 1
            
            info["message"] += "\nThe game has ended!"

        # Create Observation
        obs = {
            "dynamic_number": np.array([int(d) for d in self.dynamic_number], dtype=np.int8),
            "available_digits_per_index": self._encode_available_digits(),
            "players": self._encode_players(),
            "player_turn": np.int64(self.player_turn)
        }
        
        # Render to see output
        if self.render_mode == "human" and getattr(self, "auto_render", True):
            self.render()

        return obs, float(reward), terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            print()
            print(f"Turn: Player{self.player_turn}")
            print(f"Number: {self.dynamic_number}")
            print(f"Available digits per index: {self.available_digits_per_index}")
            print('*** Scoreboard ***')
            for p in self.players:
                print(f"Player{p['id']}: {p['score']} pts")
            print('******************')


    def close(self):
        return super().close()



if __name__ == "__main__":
    # check_env(Divide21Env())

    # initialize environment to test
    env = Divide21Env(
        digits = 2,
        players = [{"id": 0, "score": 0, "is_current_turn": 1}, {"id": 1, "score": 0, "is_current_turn": 0}],
        # render_mode='human', # comment this line when training RL models
        # auto_render=True # comment this line when training RL models
    )
    obs, info = env.reset()
    # test five examples
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        # print(info)
        if done or trunc:
            break
    
