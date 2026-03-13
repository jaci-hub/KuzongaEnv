'''
Name: Jacinto Jeje Matamba Quimua
Date: 10/28/2025

This is the python gym-style API for my game Kuzonga
'''

import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
from kuzongaenv.inspection.inspector import Inspector



class KuzongaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, digits=2, players=1, render_mode=None, auto_render=False):
        super().__init__()
        self.players = [{"i": i, "c": 0, "m": 1 if i==0 else 0} for i in range(players)]
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
            "v": spaces.Discrete(2),
            "g": spaces.Discrete(10),
            "r": spaces.Discrete(digits)
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
            "s": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "d": spaces.Box(
                low=0,
                high=9,
                shape=(digits,),
                dtype=np.int8
            ),
            "a": spaces.MultiBinary(10 * digits),
            "p": spaces.Box(
                low=np.array([0, -self.maxScore-8, 0] * number_of_players, dtype=np.int64),
                high=np.array([number_of_players - 1, self.maxScore+8, 1] * number_of_players, dtype=np.int64),
                shape=(number_of_players * 3,),
                dtype=np.int64
            ),
            "t": spaces.Discrete(number_of_players)
        })
    
    
    def _encode_players(self, given_players=None):
        '''
        Encodes player info numerically:
        Each player has attributes: i, c, m
            Note: if m=1, it is the player's turn to play, else, it is not 
        '''
        if given_players != None:
            self.players = given_players
        
        if not self.players:
            # create a default single-player representation
            encoded = np.zeros((1, 3), dtype=np.int64)
            encoded[0] = [0, 0, 1]  # i=0, c=0, m=1
            return encoded.flatten()
        
        num_players = len(self.players)
        encoded = np.zeros((num_players, 3), dtype=np.int64)
        for i, p in enumerate(self.players):
            encoded[i, 0] = p.get("i", i)
            encoded[i, 1] = p.get("c", 0)
            encoded[i, 2] = 1 if i == self.player_turn else 0
        return encoded.flatten()
    
    def _has_one_digit_divisor(self, num):
        res = False
        for i in range(2, 10):
            if num % i == 0:
                res = True
                break
        return res
    
    def _create_dynamic_number(self):
        min_val = 10 ** (self.digits - 1)
        max_val = (10 ** self.digits) - 1

        num = random.randint(min_val, max_val)

        while self._has_one_digit_divisor(num):
            num = random.randint(min_val, max_val)

        return int(num)
    
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
        if given_available_digits_per_rindex is not None:
            # convert string keys to int if necessary
            self.available_digits_per_rindex = {
                int(k): v for k, v in given_available_digits_per_rindex.items()
            }

        mask = np.zeros((self.digits, 10), dtype=np.int64)
        for idx, available in self.available_digits_per_rindex.items():
            mask[int(idx), available] = 1
        return mask.flatten()

    def reset(self, *, seed = None, options = None):
        manual_obs = None
        if options:
            manual_obs = options.get('obs', None)
        inspector = Inspector(state=manual_obs)
        inspector.inspect_state()
        
        if inspector.state_passed():
            return self._manual_reset(seed=seed, options=options)
        
        super().reset(seed=seed)
        original_number = self._create_dynamic_number()
        self.static_number = original_number
        self.dynamic_number = original_number
        self.available_digits_per_rindex = self._setup_available_digits_per_rindex()
        self.player_turn = 0
        obs = {
            "s": np.array([int(d) for d in str(self.static_number)], dtype=np.int8),
            "d": np.array([int(d) for d in str(self.dynamic_number)], dtype=np.int8),
            "a": self._encode_available_digits(),
            "p": self._encode_players(),
            "t": np.int64(self.player_turn)
        }
        
        info = {"seed": seed}
        # add decoded obs to info
        obs_decoded = self._decode_state(obs)
        info['obs_decoded'] = obs_decoded
        # add all_actions to info
        info['all_actions'] = self._get_all_actions(obs_decoded)

        return obs, info
    
    def _manual_reset(self, *, seed = None, options = None):
        '''
        resets the ennvironment manually with the obs key-value dictionary (obs) as an argument.
            if obs does not pass inspection, it falls back to the default gym-env reset
        '''
        super().reset(seed=seed)
        
        obs = None
        if options:
            obs = options.get('obs', None)
        else:
            return # It should not get here!
        
        # update observation space var
        number_of_players = len(obs["p"])
        self.maxScore = 9*len(str(obs["s"]))
        self.observation_space = spaces.Dict({
            "s": spaces.Box(
                low=0,
                high=9,
                shape=(len(str(obs["s"])),),
                dtype=np.int8
            ),
            "d": spaces.Box(
                low=0,
                high=9,
                shape=(len(str(obs["d"])),),
                dtype=np.int8
            ),
            "a": spaces.MultiBinary(10 * len(str(obs["d"]))),
            "p": spaces.Box(
                low=np.array([0, -self.maxScore-8, 0] * number_of_players, dtype=np.int64),
                high=np.array([number_of_players - 1, self.maxScore+8, 1] * number_of_players, dtype=np.int64),
                shape=(number_of_players * 3,),
                dtype=np.int64
            ),
            "t": spaces.Discrete(number_of_players)
        })
        
        # update the number of digits
        self.digits = len(str(obs["d"]))
        
        self.static_number = obs["s"]
        self.dynamic_number = obs["d"]
        # convert string keys to int if necessary
        self.available_digits_per_rindex = {
            int(k): v for k, v in obs["a"].items()
        }
        self.players = obs["p"]
        self.player_turn = obs["t"]
        
        new_obs = {
            "s": np.array([int(d) for d in str(obs["s"])], dtype=np.int8),
            "d": np.array([int(d) for d in str(obs["d"])], dtype=np.int8),
            "a": self._encode_available_digits(obs["a"]),
            "p": self._encode_players(obs["p"]),
            "t": np.int64(obs["t"])
        }
        
        info = {"seed": seed}
        info = {"manual_reset": True}
        # add decoded obs to info
        info['obs_decoded'] = obs
        # add all_actions to info
        info['all_actions'] = self._get_all_actions(obs)

        return new_obs, info
    
    def _get_all_actions(self, state):
        all_actions=[]
        # (1) add the division actions (v=1)
        for g in range(2,10):
            action={
                'v': 1,
                'g': g,
                'r': None
            }
            all_actions.append(action)
        # (2) add the digit change actions (v=0)
        for rindex in state['a']:
            available_digits = state['a'][rindex]
            for g in available_digits:
                action={
                    'v': 0,
                    'g': g,
                    'r': rindex
                }
                all_actions.append(action)
        
        return all_actions
    
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
            if player["c"] >= self.maxScore:
                return True
        # (3) only one player left without -max points or less
        count = 0
        for player in self.players:
            if player["c"] <= -self.maxScore:
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
                while self.players[self.player_turn]["c"] <= -self.maxScore:
                    self.player_turn = (self.player_turn + 1) % len(self.players)
            self.players[self.player_turn]["m"] = 1
            for player in self.players:
                if player["i"] != self.players[self.player_turn]["i"]:
                    player["m"] = 0
    
    # START | Functions to decode the state | START
    def _decode_dynamic_number(self, dynamic_number):
        decoded_dynamic_number = [str(d) for d in dynamic_number.tolist()]
        decoded_dynamic_number = ''.join(decoded_dynamic_number)
        decoded_dynamic_number = int(decoded_dynamic_number)
        return decoded_dynamic_number
    
    def _decode_available_digits(self, flat_mask, digits):
        '''
        Reconstruct the 'a' dictionary from the
        flattened mask produced by _encode_available_digits().

        Args:
            flat_mask (array-like): Flattened binary mask of shape (self.digits * 10,)

        Returns:
            dict[int, list[int]]: Dictionary mapping each rindex to its list of available digits.
        '''
        # Ensure it's a list of ints
        flat_list = list(flat_mask)
        # Rebuild mask as 2D matrix (digits x 10)
        mask_2d = [flat_list[i * 10 : (i + 1) * 10] for i in range(digits)]
        
        # Decode dictionary
        decoded = {}
        for rindex, row in enumerate(mask_2d):
            available_digits = [digit for digit, flag in enumerate(row) if flag == 1]
            decoded[rindex] = available_digits

        return decoded
    
    def _decode_players(self, flat_array):
        '''
        Reconstruct the list of player dictionaries from the flattened NumPy array
        produced by _encode_players().

        Args:
            flat_array (array-like): Flattened 1D list or NumPy array of shape (num_players * 3,).

        Returns:
            list[dict]: List of players, each as {"i": int, "c": int, "m": int}.
        '''
        # Ensure it's a plain list of ints
        flat_list = list(flat_array)
        
        # Each player has 3 attributes: [i, c, m]
        num_players = len(flat_list) // 3
        
        players = []
        for i in range(num_players):
            start = i * 3
            pid, score, turn_flag = flat_list[start:start + 3]
            player = {
                "i": int(pid),
                "c": int(score),
                "m": int(turn_flag)
            }
            players.append(player)
        
        return players
    
    def _decode_player_turn(self, player_turn):
        return int(player_turn)
    
    def _decode_state(self, state):
        '''
        the observation space attributes from KuzongaEnv are in numpy variables which are not json compatible, 
        so make sure they are.
        '''
        decoded_static_number = self._decode_dynamic_number(state["s"])
        decoded_dynamic_number = self._decode_dynamic_number(state["d"])
        decoded_state = {
            "s": decoded_static_number,
            "d": decoded_dynamic_number,
            "a": self._decode_available_digits(state["a"], len(str(decoded_dynamic_number))),
            "p": self._decode_players(state["p"]),
            "t": self._decode_player_turn(state["t"])
        }
        
        return decoded_state
    # END | Functions to decode the state | END
    
    def step(self, action):
        """
        Executes one step of the Kuzonga environment.
        Args:
            action (dict): {
                "v": 0 or 1,
                "g": int,
                "r": int if division == 0, else None
            }
        Returns:
            obs, reward, terminated, truncated, info
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # check action
        expected_keys = {"v", "g", "r"}
        if not isinstance(action, dict):
            reward += -5
            info["critical"] = "Action must be a Python dictionary."
        elif set(action.keys()) != expected_keys:
            reward += -5
            info["critical"] = f"Action dictionary must have exactly these keys: {', '.join(expected_keys)}."
        else:
            # get attributes
            division = bool(action["v"]) if action["v"] in [0, 1, True, False] else None
            digit = int(action["g"]) if action["g"] in range(0, 10) else None
            rindex = int(action["r"]) if (isinstance(action["r"], (int, np.integer)) and action["r"]>=0) else None

            # check division
            if division is None:
                reward += -5
                info["critical"] = "The value for the division key, v, must be either True or False, or 1 or 0."
            # check digit
            elif digit is None:
                reward += -5
                info["critical"] = "Digit must be between 0-9."
            # check rindex
            elif rindex is None and division is None:
                reward += -5
                info["critical"] = "Rindex, r, must be an integer greater than or equal to 0."
            
            # (1) Division attempt
            elif division:
                # deduct points if rindex is not None
                if rindex != None:
                    reward += -2
                    info["warning"] = "Rindex, r, should have not been provided!"
                
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
                        self.players[self.player_turn]["c"] += digit
                    info["note"] = f"Divided by {digit}."
                else:
                    reward += -1
                    # update player score
                    if self.players:
                        self.players[self.player_turn]["c"] -= digit
                        if self.players[self.player_turn]["c"] <= -self.maxScore:
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
                    info["note"] = f"Updated digit at rindex r={rindex} to {digit}."
                else:
                    reward += -2
                    info["warning"] = f"Cannot update the digit at rindex r={rindex} to {digit}."

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
            "s": np.array([int(d) for d in str(self.static_number)], dtype=np.int8),
            "d": np.array([int(d) for d in str(self.dynamic_number)], dtype=np.int8),
            "a": self._encode_available_digits(),
            "p": self._encode_players(),
            "t": np.int64(self.player_turn)
        }

        # add decoded obs to info
        obs_decoded = self._decode_state(obs)
        info['obs_decoded'] = obs_decoded
        # add all_actions to info
        info['all_actions'] = self._get_all_actions(obs_decoded)

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
                id = p["i"]
                score = p["c"]
                print(f"Player{id}: {score} pts")
            print('******************')


    def close(self):
        return super().close()
    