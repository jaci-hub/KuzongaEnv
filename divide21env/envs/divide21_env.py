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
import warnings
from divide21env.inspection.inspector import Inspector



class Divide21Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, digits=2, players=1, render_mode=None, auto_render=False):
        super().__init__()
        self.players = [{"id": i, "score": 0, "is_current_turn": 1 if i==0 else 0} for i in range(players)]
        self.static_number = None
        self.dynamic_number = None
        self.player_turn = 0
        self.digits = digits
        self.available_digits_per_rindex = {i: list(range(10)) for i in range(digits)}
        self.maxScore = 9*digits
        self.render_mode = render_mode
        self.auto_render = auto_render
        
        warnings.filterwarnings(
            "ignore",
            message=".*Box observation space maximum and minimum values are equal.*"
        )

        # (1) Action space:
        # action is dictionary with keys: division, digit, rindex
        #   division (bool): true/false
        #   digit (int): if division=true, then it is the divisor, else it is the new digit in the rindex chosen
        #   rindex (int): if division=true, then it is None, else the rindex where the digit will be overwriten
        self.action_space = spaces.Dict({
            "division": spaces.Discrete(2),
            "digit": spaces.Discrete(10),
            "rindex": spaces.Discrete(digits)
        })

        # (2) Observation space: 
        # observation is a dictionary with keys: static_number, dynamic_number, available_digits_per_rindex, players, player_turn]
        #   static_number (int): the value of the number originally generated
        #   dynamic_number (int): the current value of the number whose digits are manipulated
        #   available_digits_per_rindex (dict): a dictionary where the keys are the rindexes of the dynamic_number and their values are the list of digits available at that rindex
        #   players (list): the list of dictionaries with each player's id, score and a variable (is_current_turn) that tells if it is the player's turn to play. By default there is one player in the list
        #   player_turn (int): the id of the player with the turn
        number_of_players = len(self.players)
        self.observation_space = spaces.Dict({
            "static_number": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "dynamic_number": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "available_digits_per_rindex": spaces.MultiBinary(10 * digits),
            "players": spaces.Box(
                low=np.array([0, -self.maxScore-8, 0] * number_of_players, dtype=np.int64),
                high=np.array([number_of_players - 1, self.maxScore+8, 1] * number_of_players, dtype=np.int64),
                shape=(number_of_players * 3,),
                dtype=np.int64
            ),
            "player_turn": spaces.Discrete(number_of_players)
        })
    
    
    def _encode_players(self, given_players=None):
        '''
        Encodes player info numerically:
        Each player has attributes: id, score, is_current_turn
            Note: if is_current_turn=1, it is the player's turn to play, else, it is not 
        '''
        if given_players != None:
            self.players = given_players
        
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
                return int(s)

        # fallback: deterministic string like "10...01"
        fallback = "1" + ("0" * (self.digits - 2)) + "1"
        return int(fallback)
    
    def _get_prohibited_digit_list_at_rindex(self, rindex):
        prohibited = set()
        # no leading zero
        if rindex == self.digits-1:
            prohibited.add(0)

        # can’t make number 0 or 1
        for d in [0, 1]:
            modified = str(self.dynamic_number)
            modified = list(modified)        
            if rindex > 0:
                modified[-rindex-1] = str(d)
            else:
                modified[self.digits-1] = str(d)
            modified = ''.join(modified)
            
            if int(modified) in (0, 1):
                prohibited.add(d)

        return list(prohibited)
    
    def _remove_all_prohibited_digits_at_given_rindex_from_given_list(self, rindex, digit_list):
        prohibited_digits = set(self._get_prohibited_digit_list_at_rindex(rindex))
        return [d for d in digit_list if d not in prohibited_digits]
    
    def _setup_available_digits_per_rindex(self):
        available_digits_per_rindex = {}

        for i in range(self.digits):
            current_digit = int(str(self.dynamic_number)[self.digits-i-1])
            all_digits = [d for d in range(10) if d != current_digit]
            filtered_digits = self._remove_all_prohibited_digits_at_given_rindex_from_given_list(i, all_digits)
            available_digits_per_rindex[i] = filtered_digits

        return available_digits_per_rindex
    
    def _encode_available_digits(self, given_available_digits_per_rindex=None):
        if given_available_digits_per_rindex != None:
            self.available_digits_per_rindex = given_available_digits_per_rindex
        
        mask = np.zeros((self.digits, 10), dtype=np.int64)
        for idx, available in self.available_digits_per_rindex.items():
            mask[idx, available] = 1
        return mask.flatten()

    def reset(self, *, seed = None, options = None, manual_obs=None):
        inspector = Inspector(state=manual_obs)
        inspector.inspect_state()
        
        if inspector.state_passed():
            return self._manual_reset(seed=seed, obs=manual_obs)
        
        super().reset(seed=seed)
        original_number = self._create_dynamic_number()
        self.static_number = original_number
        self.dynamic_number = original_number
        self.available_digits_per_rindex = self._setup_available_digits_per_rindex()
        self.player_turn = 0
        obs = {
            "static_number": np.array([int(d) for d in str(self.static_number)], dtype=np.int8),
            "dynamic_number": np.array([int(d) for d in str(self.dynamic_number)], dtype=np.int8),
            "available_digits_per_rindex": self._encode_available_digits(),
            "players": self._encode_players(),
            "player_turn": np.int64(self.player_turn)
        }
        
        info = {"seed": seed}
        return obs, info
    
    def _manual_reset(self, *, seed = None, options = None, obs = None):
        '''
        resets the ennvironment manually with the obs key-value dictionary (obs) as an argument.
            if obs does not pass inspection, it falls back to the default gym-env reset
        '''
        super().reset(seed=seed)
        
        # update observation space var
        digits = len(str(obs["static_number"]))
        number_of_players = len(obs["players"])
        self.maxScore = 9*digits
        self.observation_space = spaces.Dict({
            "static_number": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "dynamic_number": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "available_digits_per_rindex": spaces.MultiBinary(10 * digits),
            "players": spaces.Box(
                low=np.array([0, -self.maxScore-8, 0] * number_of_players, dtype=np.int64),
                high=np.array([number_of_players - 1, self.maxScore+8, 1] * number_of_players, dtype=np.int64),
                shape=(number_of_players * 3,),
                dtype=np.int64
            ),
            "player_turn": spaces.Discrete(number_of_players)
        })
        
        self.static_number = obs["static_number"]
        self.dynamic_number = obs["dynamic_number"]
        self.available_digits_per_rindex = obs["available_digits_per_rindex"]
        self.players = obs["players"]
        self.player_turn = obs["player_turn"]
        
        new_obs = {
            "static_number": np.array([int(d) for d in str(obs["static_number"])], dtype=np.int8),
            "dynamic_number": np.array([int(d) for d in str(obs["dynamic_number"])], dtype=np.int8),
            "available_digits_per_rindex": self._encode_available_digits(obs["available_digits_per_rindex"]),
            "players": self._encode_players(obs["players"]),
            "player_turn": np.int64(obs["player_turn"])
        }
        
        info = {"seed": seed}
        info = {"manual_reset": True}
        return new_obs, info
    
    def _rindex_available_digit_list_is_empty(self, rindex):
        if not self.available_digits_per_rindex:
            return True

        rindex_available_digit_list = self.available_digits_per_rindex[rindex]
        return len(rindex_available_digit_list) == 0
    
    def _update_available_digits_per_rindex(self, rindex=None):
        if rindex is None: # division was performed
            for i in range(self.digits):
                current_digit = int(str(self.dynamic_number)[self.digits-i-1])
                rindex_available_digit_list = [d for d in self.available_digits_per_rindex[i] if d != current_digit]
                rindex_available_digit_list = self._remove_all_prohibited_digits_at_given_rindex_from_given_list(i, rindex_available_digit_list)

                if rindex_available_digit_list:
                    self.available_digits_per_rindex[i] = rindex_available_digit_list
                else:
                    all_digits = [d for d in range(10) if d != current_digit]
                    all_digits = self._remove_all_prohibited_digits_at_given_rindex_from_given_list(i, all_digits)
                    self.available_digits_per_rindex[i] = all_digits
        else: # division was not performed
            if not self._rindex_available_digit_list_is_empty(rindex):
                return
            
            current_digit = None
            if rindex>0:
                current_digit = int(str(self.dynamic_number)[-rindex-1])
            else:
                current_digit = int(str(self.dynamic_number)[self.digits-1])
            all_digits = [d for d in range(10) if d != current_digit]
            all_digits = self._remove_all_prohibited_digits_at_given_rindex_from_given_list(rindex, all_digits)
            self.available_digits_per_rindex[rindex] = all_digits
    
    def _remove_each_quotient_digit_from_available_digits_per_rindex(self, quotient_string):
        if not self.available_digits_per_rindex:
            return

        for i in range(len(quotient_string)):
            rindex_available_digit_list = self.available_digits_per_rindex[i]
            if not rindex_available_digit_list:
                continue

            digit_to_remove = int(quotient_string[self.digits-i-1])
            rindex_available_digit_list = [d for d in rindex_available_digit_list if d != digit_to_remove]
            self.available_digits_per_rindex[i] = rindex_available_digit_list
    
    def _remove_digit_from_rindex_available_digits(self, rindex, digit_to_remove):
        if not self.available_digits_per_rindex:
            return

        rindex_available_digit_list = self.available_digits_per_rindex[rindex]
        if not rindex_available_digit_list:
            return

        rindex_available_digit_list = [d for d in rindex_available_digit_list if d != digit_to_remove]
        self.available_digits_per_rindex[rindex] = rindex_available_digit_list
    
    def _game_over(self):
        # (1) quotient 1
        if self.dynamic_number == 1:
            return True
        # (2) max points
        for player in self.players:
            if player['score'] >= self.maxScore:
                return True
        # (3) only one player left without -max points or less
        count = 0
        for player in self.players:
            if player['score'] <= -self.maxScore:
                if len(self.players) > 1:
                    count += 1
                else:
                    return True
        if len(self.players) > 1 and count == len(self.players) - 1:
            return True
        
        return False
    
    def _update_player_turn(self):
        if self.players:
            self.player_turn = (self.player_turn + 1) % len(self.players)
            if len(self.players) > 1:
                while self.players[self.player_turn]['score'] <= -self.maxScore:
                    self.player_turn = (self.player_turn + 1) % len(self.players)
            self.players[self.player_turn]['is_current_turn'] = 1
            for player in self.players:
                if player['id'] != self.players[self.player_turn]['id']:
                    player['is_current_turn'] = 0
    
    def step(self, action):
        """
        Executes one step of the Divide21 environment.
        Args:
            action (dict): {
                "division": 0 or 1,
                "digit": int,
                "rindex": int if division == 0, else None
            }
        Returns:
            obs, reward, terminated, truncated, info
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # check action
        expected_keys = {'division', 'digit', 'rindex'}
        if not isinstance(action, dict):
            reward += -5
            info["critical"] = "Action must be a Python dictionary."
        elif set(action.keys()) != expected_keys:
            reward += -5
            info["critical"] = f"Action dictionary must have exactly these keys: {', '.join(expected_keys)}."
        else:
            # get attributes
            division = bool(action["division"]) if action["division"] in [0, 1, True, False] else None
            digit = int(action["digit"]) if action["digit"] in range(0, 10) else None
            rindex = int(action["rindex"]) if (isinstance(action["rindex"], (int, np.integer)) and action["rindex"]>=0) else None

            # check division
            if division is None:
                reward += -5
                info["critical"] = "The value for the division attribute must be either True or False, or 1 or 0."
            # check digit
            elif digit is None:
                reward += -5
                info["critical"] = "Digit must be between 0-9."
            # check rindex
            elif rindex is None and division is None:
                reward += -5
                info["critical"] = "Rindex must be an integer greater than or equal to 0."
            
            # (1) Division attempt
            elif division:
                # deduct points if rindex is not None
                if rindex != None:
                    reward += -2
                    info["warning"] = "Rindex should have not been provided!"
                
                if digit in [0, 1]: # not allowed to divide by 0 or 1
                    reward += -5
                    info["critical"] = "Division by 0 or 1 is not allowed!"
                elif self.dynamic_number % digit == 0:
                    self.dynamic_number = self.dynamic_number // digit
                    # whenever the number of digits in the quotient is less than that of the original number, 
                    #   remove the rindex key greater than the number of digits in the quotient
                    for j in range(len(str(self.dynamic_number)), self.digits):
                        if j in self.available_digits_per_rindex:
                            del self.available_digits_per_rindex[j]
                    # update the number of digits
                    self.digits = len(str(self.dynamic_number))
                    reward += 1
                    # update the list of available digits per rindex
                    #   (1) remove each quotient digit from available digits per rindex
                    self._remove_each_quotient_digit_from_available_digits_per_rindex(str(self.dynamic_number))
                    #   (2) update available digits per rindex
                    self._update_available_digits_per_rindex() # no need to pass the rindex, because a division was performed
                    # update player score
                    if self.players:
                        self.players[self.player_turn]["score"] += digit
                    info["note"] = f"Divided by {digit}."
                else:
                    reward += -1
                    # update player score
                    if self.players:
                        self.players[self.player_turn]["score"] -= digit
                        if self.players[self.player_turn]['score'] <= -self.maxScore:
                            # update player turn
                            self._update_player_turn()
                    info["note"] = f"Careful, {digit} is not a factor of {self.dynamic_number}."
            # (2) Digit change
            else:
                if rindex in self.available_digits_per_rindex and digit in self.available_digits_per_rindex[rindex]:
                    num_str = list(str(self.dynamic_number))
                    if rindex>0:
                        num_str[-rindex-1] = str(digit)
                    else:
                        num_str[len(num_str)-1] = str(digit)
                    self.dynamic_number = "".join(num_str)
                    self.dynamic_number = int(self.dynamic_number)
                    reward += 1
                    # update the list of available digits per rindex
                    # (1) remove digit from rindex available digits
                    self._remove_digit_from_rindex_available_digits(rindex, digit)
                    # (2) update available digits per rindex
                    self._update_available_digits_per_rindex(rindex)
                    # update player turn
                    self._update_player_turn()
                    info["note"] = f"Updated digit at rindex {rindex} to {digit}."
                else:
                    reward += -2
                    info["warning"] = f"Cannot update the digit at rindex {rindex} to {digit}."

        # Check if game is over
        if self._game_over():
            terminated = True
            
            if reward > 0:
                reward += 10
            else:
                reward -= 10
            
            info["concluded"] = True

        # Create Observation
        obs = {
            "static_number": np.array([int(d) for d in str(self.static_number)], dtype=np.int8),
            "dynamic_number": np.array([int(d) for d in str(self.dynamic_number)], dtype=np.int8),
            "available_digits_per_rindex": self._encode_available_digits(),
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
            print(f"Static Number: {self.static_number}")
            print(f"Dynamic Number: {self.dynamic_number}")
            print(f"Available digits per rindex: {self.available_digits_per_rindex}")
            print(f"Turn: Player{self.player_turn}")
            print('*** Scoreboard ***')
            for p in self.players:
                print(f"Player{p['id']}: {p['score']} pts")
            print('******************')


    def close(self):
        return super().close()
    