"""Groupier functions"""
import logging
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Sequence
# instead of from gymnasium.vector.utils.spaces import Dict, use spaces. is a less ambiguous way
from gymnasium.vector.utils import spaces

from agents.player_interface import Player
from gym_env.cycle import PlayerCycle
from gym_env.enums import Action, Stage
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from tools.hand_evaluator import get_winner
from tools.montecarlo_python import get_equity

# pylint: disable=import-outside-toplevel

log = logging.getLogger(__name__)

winner_in_episodes = []
MONTEACRLO_RUNS = 1000  # relevant for equity calculation if switched on


def community_spec_generator(num_players: int) -> spaces.Dict:
    return spaces.Dict({
        "current_player_position": spaces.MultiBinary(num_players),
        "stage": spaces.MultiBinary(4),
        "community_pot": spaces.Box(low=0, high=np.inf, shape=(1,)),
        "current_round_pot": spaces.Box(low=0, high=np.inf, shape=(1,)),
        "active_players": spaces.MultiBinary(num_players),
        "big_blind": spaces.Box(low=0, high=np.inf, shape=(1,)),
        "small_blind": spaces.Box(low=0, high=np.inf, shape=(1,)),
        "legal_moves": spaces.MultiBinary(len(Action))
    })


def community_data_init(num_players: int) -> dict:
    return {
        "current_player_position": [False] * num_players,
        "stage": [False] * 4,
        "community_pot": 0,
        "current_round_pot": 0,
        "active_players": [False] * num_players,
        "big_blind": 0,
        "small_blind": 0,
        "legal_moves": [0 for _ in Action]
    }


def stage_spec_generator(num_players: int) -> spaces.Tuple:
    return spaces.Tuple([spaces.Dict({
        "calls": spaces.MultiBinary(num_players),
        "raises": spaces.MultiBinary(num_players),
        "min_call_at_action": spaces.Box(low=0, high=np.inf, shape=(num_players,)),
        "contribution": spaces.Box(low=0, high=np.inf, shape=(num_players,)),
        "stack_at_action": spaces.Box(low=0, high=np.inf, shape=(num_players,)),
        "community_pot_at_action": spaces.Box(low=0, high=np.inf, shape=(num_players,))
    }) for _ in range(8)])


def stage_data_init(num_players: int) -> List[dict]:
    return [{
        "calls": np.zeros(num_players),
        "raises": np.zeros(num_players),
        "min_call_at_action": np.zeros(num_players),
        "contribution": np.zeros(num_players),
        "stack_at_action": np.zeros(num_players),
        "community_pot_at_action": np.zeros(num_players)
    } for _ in range(8)]


def player_spec_generator(num_players: int) -> spaces.Dict:
    return spaces.Dict({
        "position": spaces.Discrete(num_players),
        "equity_to_river_alive": spaces.Box(low=0, high=1, shape=(1,)),
        "equity_to_river_2plr": spaces.Box(low=0, high=1, shape=(1,)),
        "equity_to_river_3plr": spaces.Box(low=0, high=1, shape=(1,)),
        "stack": spaces.Box(low=0, high=np.inf, shape=(1,))
    })


def player_init(initial_stacks_per_bb: int = 100) -> dict:
    return {
        "position": 0,
        "equity_to_river_alive": 0,
        "equity_to_river_2plr": 0,
        "equity_to_river_3plr": 0,
        "stack": initial_stacks_per_bb
    }


def observation_init(num_players: int, big_blind: int, small_blind: int) -> dict:
    observation = {
        "players": [player_init() for _ in range(num_players)],
        "community_data": community_data_init(num_players),
        "stage_data": stage_data_init(num_players)
    }
    observation['community_data']['big_blind'] = big_blind
    observation['community_data']['small_blind'] = small_blind
    return observation


class HoldemTable(Env):
    """Pokergame environment"""

    def __init__(self, initial_stacks=100, small_blind=1, big_blind=2, render=False, funds_plot=True,
            max_raises_per_player_round=2, players: List[Player] = [], raise_illegal_moves=False,
            calculate_equity=False):
        """
        The table needs to be initialized once at the beginning

        Args:
            initial_stacks (real): initial stacks per player
            small_blind (real)
            big_blind (real)
            render (bool): render table after each move in graphical format
            funds_plot (bool): show plot of funds history at end of each episode
            max_raises_per_player_round (int): max raises per round per player

        """
        assert len(players) > 1, "At least two players are needed"

        self.num_of_players = len(players)
        self.get_equity = get_equity
        self.observation_space = spaces.Dict({
            "players": player_spec_generator(len(players)),
            "community_data": community_spec_generator(len(players)),
            "stage_data": stage_spec_generator(len(players))
        })
        self.action_space = Discrete(len(Action) - 2)
        self.observation = observation_init(len(players), big_blind, small_blind)

        # game info, but not in observation
        self.initial_stacks = initial_stacks
        # todo: might use mask to hide the other player's data
        self.players = []
        for idx, player in enumerate(players):
            self._add_player(player)

        # misc
        self.render_switch = render

        # init in start new hand
        self.table_cards = []
        self.dealer_pos = None
        self.player_status = []  # one hot encoded
        self.player_cycle = None  # cycle iterator
        self.last_player_pot = None
        self.viewer = None
        self.player_max_win = None  # used for side pots
        self.round_number_in_street = 0
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.played_in_round = None
        self.min_call = None
        self.community_data = None
        self.deck = None
        self.action = None
        self.winner_ix = None
        self.acting_agent = None
        self.funds_plot = funds_plot
        self.max_raises_per_player_round = max_raises_per_player_round
        self.calculate_equity = calculate_equity

        # pots
        self.community_pot = 0
        self.current_round_pot = 9
        self.player_pots = None  # individual player pots

        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = None
        self.legal_moves = None
        self.illegal_move_reward = -1
        self.action_space = Discrete(len(Action) - 2)
        self.first_action_for_hand = None
        self.raise_illegal_moves = raise_illegal_moves

        self.dealer_pos = 0
        self.reset()

    # todo: might have way to remove self
    def reset(self) -> Dict:
        """Reset after game over."""

        # some value might need to keep in whole game(env)
        self.observation = observation_init(len(self.players), self.observation['community_data']['big_blind'],
                                            self.observation['community_data']['small_blind'])
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = pd.DataFrame()
        # Generate inital hands
        self.first_action_for_hand = [True] * len(self.players)

        if not self.players:
            log.warning("No agents added. Add agents before resetting the environment.")
            return

        for player in self.players:
            player.stack = self.initial_stacks

        self.dealer_pos = 0
        max_steps_after_raiser = (self.max_raises_per_player_round - 1) * len(self.players) - 1
        self.player_cycle = PlayerCycle(self.players, dealer_idx=-1, max_steps_after_raiser=max_steps_after_raiser,
                                        max_steps_after_big_blind=len(self.players),
                                        max_raises_per_player_round=self.max_raises_per_player_round)
        self._start_new_hand()
        self._get_environment()
        # auto play for agents where autoplay is set
        if self._is_agent_autoplay() and not self.done:
            self.step('initial_player_autoplay')  # kick off the first action after bb by an autoplay agent
        # return real observation
        return self.observation

    # Step is too complicated compare with other gym example. It's better to split it into smaller functions
    def step(self, action):  # pylint: disable=arguments-differ
        """
        Next player makes a move and a new environment is observed.

        Args:
            action: Used for testing only. Needs to be of Action type

        """
        # loop over step function, calling the agent's action method
        # until either the env id done, or an agent is just a shell and
        # and will get a call from to the step function externally (e.g. via
        # keras-rl
        self.reward = 0
        self.acting_agent = self.player_cycle.idx
        # todo: 我感覺是區分不同屬性的autoplay agent
        if self._is_agent_autoplay():
            while self._is_agent_autoplay() and not self.done:
                log.debug("Autoplay agent. Call action method of agent.")
                self._get_environment()
                # call agent's action method
                action = self.current_player.agent_obj.action(self.legal_moves, self.observation, self.info)
                if Action(action) not in self.legal_moves:
                    self._illegal_move(action)
                else:
                    self._execute_step(Action(action))
                    if self.first_action_for_hand[self.acting_agent] or self.done:
                        self.first_action_for_hand[self.acting_agent] = False
                        self._calculate_reward(action)

        else:  # action received from player shell (e.g. keras rl, not autoplay)
            self._get_environment()  # get legal moves
            if Action(action) not in self.legal_moves:
                self._illegal_move(action)
            else:
                self._execute_step(Action(action))
                if self.first_action_for_hand[self.acting_agent] or self.done:
                    self.first_action_for_hand[self.acting_agent] = False
                    self._calculate_reward(action)

            log.debug(f"Previous action reward for seat {self.acting_agent}: {self.reward}")
        # have return observe, reward , done, info
        # todo: implement array_everythin.
        return self.observation, self.reward, self.done, self.info

    def _execute_step(self, action):
        self._process_decision(action)

        self._next_player()

        if self.observation['community_data']['stage'] in [Stage.END_HIDDEN, Stage.SHOWDOWN]:
            self._end_hand()
            self._start_new_hand()

        self._get_environment()

    def _illegal_move(self, action):
        log.warning(f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}")
        if self.raise_illegal_moves:
            raise ValueError(f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}")
        self.reward = self.illegal_move_reward

    def _is_agent_autoplay(self, idx=None):
        # todo: idx 是啥？換到誰了？
        if not idx:
            return hasattr(self.current_player.agent_obj, 'autoplay')
        return hasattr(self.players[idx].agent_obj, 'autoplay')

    # todo: Might not need this. It's
    def _get_environment(self) -> None:
        """
        Update all env data. In this env. The players data should be updated, however, only display the current player on observation.
        todo: Might figure out better way to disclose the player data( mayby use mask)
        """
        if not self.done:
            self._get_legal_moves()

        self.reward = 0
        self.info = None

        # update all observation
        self.observation['community_data']['community_pot'] = self.community_pot / (self.observation['community_data']['big_blind'] * 100)
        self.observation['community_data']['current_round_pot'] = self.current_round_pot / (self.observation['community_data']['big_blind'] * 100)
        current_stage = np.argmax(self.observation['community_data']['stage'])
        self.observation['community_data']['stage'][min(current_stage, 3)] = True
        self.observation['community_data']['legal_moves'] = [action in self.legal_moves for action in Action]


        if not self.current_player:  # game over
            self.current_player = self.players[self.winner_ix]

        current_player = self.observation['players'][self.current_player]
        if self.calculate_equity:
            current_player.equity_alive = self.get_equity(set(self.current_player.cards), set(self.table_cards),
                                                               sum(self.player_cycle.alive), MONTEACRLO_RUNS)
            current_player.equity_to_river_2plr = self.get_equity(set(self.current_player.cards),
                                                                    set(self.table_cards),
                                                                    sum(self.player_cycle.alive), MONTEACRLO_RUNS)
            current_player.equity_to_river_3plr = self.get_equity(set(self.current_player.cards),
                                                                    set(self.table_cards),
                                                                    sum(self.player_cycle.alive), MONTEACRLO_RUNS)
        else:
            current_player.equity_alive = np.nan
            current_player.equity_to_river_2plr = np.nan
            current_player.equity_to_river_3plr = np.nan

        self.current_player.equity_alive = self.get_equity(set(self.current_player.cards), set(self.table_cards),
                                                           sum(self.player_cycle.alive), 1000)
        self.current_player.equity_to_river_alive = self.current_player.equity_alive

        self._get_legal_moves()

        self.info = {'legal_moves': self.legal_moves}

        if self.render_switch:
            self.render()

    def _calculate_reward(self, last_action):
        """
        Preliminiary implementation of reward function

        - Currently missing potential additional winnings from future contributions
        """
        # if last_action == Action.FOLD:
        #     self.reward = -(
        #             self.community_pot + self.current_round_pot)
        # else:
        #     self.reward = self.player_data.equity_to_river_alive * (self.community_pot + self.current_round_pot) - \
        #                   (1 - self.player_data.equity_to_river_alive) * self.player_pots[self.current_player.seat]
        _ = last_action
        if self.done:
            # todo: 只有這裡 agent autoplay才會有 winner_idx 這個屬性
            won = 1 if not self._is_agent_autoplay(idx=self.winner_ix) else -1
            self.reward = self.initial_stacks * len(self.players) * won
            log.debug(f"Keras-rl agent has reward {self.reward}")

        elif len(self.funds_history) > 1:
            self.reward = self.funds_history.iloc[-1, self.acting_agent] - self.funds_history.iloc[
                -2, self.acting_agent]

        else:
            pass

    def _process_decision(self, action):  # pylint: disable=too-many-statements
        """Process the decisions that have been made by an agent."""
        if action not in [Action.SMALL_BLIND, Action.BIG_BLIND]:
            assert action in set(self.legal_moves), "Illegal decision"

        if action == Action.FOLD:
            self.player_cycle.deactivate_current()
            self.player_cycle.mark_folder()

        else:

            if action == Action.CALL:
                contribution = min(self.min_call - self.player_pots[self.current_player.seat],
                                   self.current_player.stack)
                self.callers.append(self.current_player.seat)
                self.last_caller = self.current_player.seat

            # verify the player has enough in his stack
            elif action == Action.CHECK:
                contribution = 0
                self.player_cycle.mark_checker()

            elif action == Action.RAISE_3BB:
                contribution = 3 * self.observation['community_data']['big_blind'] - self.player_pots[self.current_player.seat]
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1

            elif action == Action.RAISE_HALF_POT:
                contribution = (self.community_pot + self.current_round_pot) / 2
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1

            elif action == Action.RAISE_POT:
                contribution = (self.community_pot + self.current_round_pot)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1

            elif action == Action.RAISE_2POT:
                contribution = (self.community_pot + self.current_round_pot) * 2
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1

            elif action == Action.ALL_IN:
                contribution = self.current_player.stack
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1

            elif action == Action.SMALL_BLIND:
                contribution = np.minimum(self.observation['community_data']['small_blind'], self.current_player.stack)


            elif action == Action.BIG_BLIND:
                contribution = np.minimum(self.observation['community_data']['big_blind'], self.current_player.stack)
                self.player_cycle.mark_bb()
            else:
                raise RuntimeError("Illegal action.")

            if contribution > self.min_call and not (action == Action.BIG_BLIND or action == Action.SMALL_BLIND):
                self.player_cycle.mark_raiser()

            self.current_player.stack -= contribution
            self.player_pots[self.current_player.seat] += contribution
            self.current_round_pot += contribution
            self.last_player_pot = self.player_pots[self.current_player.seat]

            if self.current_player.stack == 0 and contribution > 0:
                self.player_cycle.mark_out_of_cash_but_contributed()

            self.min_call = max(self.min_call, contribution)

            self.current_player.actions.append(action)
            self.current_player.last_action_in_stage = action.name
            self.current_player.temp_stack.append(self.current_player.stack)

            self.player_max_win[self.current_player.seat] += contribution  # side pot

            pos = self.player_cycle.idx
            rnd = self.stage.value + self.round_number_in_street
            self.observation['stage_data'][rnd]['calls'][pos] = action == Action.CALL
            self.observation['stage_data'][rnd]['raises'][pos] = action in [Action.RAISE_2POT, Action.RAISE_HALF_POT, Action.RAISE_POT]
            self.observation['stage_data'][rnd]['min_call_at_action'][pos] = self.min_call / (self.observation['community_data']['big_blind'] * 100)
            self.observation['stage_data'][rnd]['community_pot_at_action'][pos] = self.community_pot / (
                    self.observation['community_data']['big_blind'] * 100)
            self.observation['stage_data'][rnd]['contribution'][pos] += contribution / (self.observation['community_data']['big_blind'] * 100)
            self.observation['stage_data'][rnd]['stack_at_action'][pos] = self.current_player.stack / (
                    self.observation['community_data']['big_blind'] * 100)

        self.player_cycle.update_alive()

        log.info(
            f"Seat {self.current_player.seat} ({self.current_player.name}): {action} - Remaining stack: {self.current_player.stack}, "
            f"Round pot: {self.current_round_pot}, Community pot: {self.community_pot}, "
            f"player pot: {self.player_pots[self.current_player.seat]}")

    def _start_new_hand(self):
        """Deal new cards to players and reset table states."""
        self._save_funds_history()

        if self._check_game_over():
            return

        log.info("")
        log.info("++++++++++++++++++")
        log.info("Starting new hand.")
        log.info("++++++++++++++++++")
        self.table_cards = []
        self._create_card_deck()
        self.stage = Stage.PREFLOP

        # preflop round1,2, flop>: round 1,2, turn etc...
        self.stage_data = stage_data_init(len(self.players))

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)
        self.player_max_win = [0] * len(self.players)
        self.last_player_pot = 0
        self.played_in_round = 0
        self.first_action_for_hand = [True] * len(self.players)

        for player in self.players:
            player.cards = []

        self._next_dealer()

        self._distribute_cards()
        self._initiate_round()

    def _save_funds_history(self):
        """Keep track of player funds history"""
        funds_dict = {i: player.stack for i, player in enumerate(self.players)}
        self.funds_history = pd.concat([self.funds_history, pd.DataFrame(funds_dict, index=[0])])

    def _check_game_over(self):
        """Check if only one player has money left"""
        player_alive = []
        self.player_cycle.new_hand_reset()

        for idx, player in enumerate(self.players):
            if player.stack > 0:
                player_alive.append(True)
            else:
                self.player_status.append(False)
                self.player_cycle.deactivate_player(idx)

        remaining_players = sum(player_alive)
        if remaining_players < 2:
            self._game_over()
            return True
        return False

    def _game_over(self):
        """End of an episode."""
        log.info("Game over.")
        self.done = True
        player_names = [f"{i} - {player.name}" for i, player in enumerate(self.players)]
        self.funds_history.columns = player_names
        if self.funds_plot:
            self.funds_history.reset_index(drop=True).plot()
        log.info(self.funds_history)
        plt.show()

        winner_in_episodes.append(self.winner_ix)
        league_table = pd.Series(winner_in_episodes).value_counts()
        best_player = league_table.index[0]
        log.info(league_table)
        log.info(f"Best Player: {best_player}")

    def _initiate_round(self):
        """A new round (flop, turn, river) is initiated"""
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.min_call = 0
        for player in self.players:
            player.last_action_in_stage = ''
        self.player_cycle.new_street_reset()

        # advance headsup players by 1 step after preflop
        if self.stage != Stage.PREFLOP and self.num_of_players == 2:
            self.player_cycle.idx += 1

        if self.stage == Stage.PREFLOP:
            log.info("")
            log.info("===Round: Stage: PREFLOP")
            # max steps total will be adjusted again at bb
            self.player_cycle.max_steps_total = len(self.players) * self.max_raises_per_player_round + 2

            self._next_player()
            self._process_decision(Action.SMALL_BLIND)
            self._next_player()
            self._process_decision(Action.BIG_BLIND)
            self._next_player()

        elif self.stage in [Stage.FLOP, Stage.TURN, Stage.RIVER]:
            self.player_cycle.max_steps_total = len(self.players) * self.max_raises_per_player_round

            self._next_player()

        elif self.stage == Stage.SHOWDOWN:
            log.info("Showdown")

        else:
            raise RuntimeError()

    def _add_player(self, agent):
        """Add a player to the table. Has to happen at the very beginning"""
        self.num_of_players += 1
        player = PlayerShell(stack_size=self.initial_stacks, name=agent.name)
        player.agent_obj = agent
        player.seat = len(self.players)  # assign next seat number to player
        player.stack = self.initial_stacks
        self.players.append(player)
        self.player_status = [True] * len(self.players)
        self.player_pots = [0] * len(self.players)

    def _end_round(self):
        """End of preflop, flop, turn or river"""
        self._close_round()
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self._distribute_cards_to_table(3)

        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self._distribute_cards_to_table(1)

        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self._distribute_cards_to_table(1)

        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN

        log.info("--------------------------------")
        log.info(f"===ROUND: {self.stage} ===")
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)

    def _end_hand(self):
        self._clean_up_pots()
        self.winner_ix = self._get_winner()
        self._award_winner(self.winner_ix)

    def _get_winner(self):
        """Determine which player has won the hand"""
        potential_winners = self.player_cycle.get_potential_winners()

        potential_winner_idx = [i for i, potential_winner in enumerate(potential_winners) if potential_winner]
        if sum(potential_winners) == 1:
            winner_ix = [i for i, active in enumerate(potential_winners) if active][0]
            winning_card_type = 'Only remaining player in round'

        else:
            assert self.stage == Stage.SHOWDOWN
            remaining_player_winner_ix, winning_card_type = get_winner([player.cards
                                                                        for ix, player in enumerate(self.players) if
                                                                        potential_winners[ix]],
                                                                       self.table_cards)
            winner_ix = potential_winner_idx[remaining_player_winner_ix]
        log.info(f"Player {winner_ix} won: {winning_card_type}")
        return winner_ix

    def _award_winner(self, winner_ix):
        """Hand the pot to the winner and handle side pots"""
        max_win_per_player_for_winner = self.player_max_win[winner_ix]
        total_winnings = sum(np.minimum(max_win_per_player_for_winner, self.player_max_win))
        remains = np.maximum(0, np.array(self.player_max_win) - max_win_per_player_for_winner)  # to be returned

        self.players[winner_ix].stack += total_winnings
        self.winner_ix = winner_ix
        if total_winnings < sum(self.player_max_win):
            log.info("Returning side pots")
            for i, player in enumerate(self.players):
                player.stack += remains[i]

    def _next_dealer(self):
        self.dealer_pos = self.player_cycle.next_dealer().seat

    def _next_player(self):
        """Move to the next player"""
        self.current_player = self.player_cycle.next_player()
        if not self.current_player:
            if sum(self.player_cycle.alive) < 2:
                log.info("Only one player remaining in round")
                self.stage = Stage.END_HIDDEN
            else:
                log.info("End round - no current player returned")
                self._end_round()
                # todo: in some cases no new round should be initialized bc only one player is playing only it seems
                self._initiate_round()

        elif self.current_player == 'max_steps_total' or self.current_player == 'max_steps_after_raiser':
            log.debug(self.current_player)
            log.info("End of round ")
            self._end_round()
            return

    def _get_legal_moves(self):
        """Determine what moves are allowed in the current state"""
        self.legal_moves = []
        if self.player_pots[self.current_player.seat] == max(self.player_pots):
            self.legal_moves.append(Action.CHECK)
        else:
            self.legal_moves.append(Action.CALL)
            self.legal_moves.append(Action.FOLD)

        if self.current_player.num_raises_in_street[self.stage] < self.max_raises_per_player_round:
            if self.current_player.stack >= 3 * self.observation['community_data']['big_blind'] - self.player_pots[self.current_player.seat]:
                self.legal_moves.append(Action.RAISE_3BB)

            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) / 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_HALF_POT)

            if self.current_player.stack >= (self.community_pot + self.current_round_pot) >= self.min_call:
                self.legal_moves.append(Action.RAISE_POT)

            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) * 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_2POT)

            if self.current_player.stack > 0:
                self.legal_moves.append(Action.ALL_IN)

        log.debug(f"Community+current round pot pot: {self.community_pot + self.current_round_pot}")

    def _close_round(self):
        """put player_pots into community pots"""
        self.community_pot += sum(self.player_pots)
        self.player_pots = [0] * len(self.players)
        self.played_in_round = 0

    def _create_card_deck(self):
        values = "23456789TJQKA"
        suites = "CDHS"
        self.deck = []  # contains cards in the deck
        _ = [self.deck.append(x + y) for x in values for y in suites]

    def _distribute_cards(self):
        log.info(f"Dealer is at position {self.dealer_pos}")
        for player in self.players:
            player.cards = []
            if player.stack <= 0:
                continue
            for _ in range(2):
                card = np.random.randint(0, len(self.deck))
                player.cards.append(self.deck.pop(card))
            log.info(f"Player {player.seat} got {player.cards} and ${player.stack}")

    def _distribute_cards_to_table(self, amount_of_cards):
        for _ in range(amount_of_cards):
            card = np.random.randint(0, len(self.deck))
            self.table_cards.append(self.deck.pop(card))
        log.info(f"Cards on table: {self.table_cards}")

    def render(self, mode='human'):
        """Render the current state"""
        screen_width = 600
        screen_height = 400
        table_radius = 200
        face_radius = 10

        if self.viewer is None:
            self.viewer = PygletWindow(screen_width + 50, screen_height + 50)
        self.viewer.reset()
        self.viewer.circle(screen_width / 2, screen_height / 2, table_radius, color=BLUE,
                           thickness=0)

        for i in range(len(self.players)):
            degrees = i * (360 / len(self.players))
            radian = (degrees * (np.pi / 180))
            x = (face_radius + table_radius) * np.cos(radian) + screen_width / 2
            y = (face_radius + table_radius) * np.sin(radian) + screen_height / 2
            if self.player_cycle.alive[i]:
                color = GREEN
            else:
                color = RED
            self.viewer.circle(x, y, face_radius, color=color, thickness=2)

            try:
                if i == self.current_player.seat:
                    self.viewer.rectangle(x - 60, y, 150, -50, (255, 0, 0, 10))
            except AttributeError:
                pass
            self.viewer.text(f"{self.players[i].name}", x - 60, y - 15,
                             font_size=10,
                             color=WHITE)
            self.viewer.text(f"Player {self.players[i].seat}: {self.players[i].cards}", x - 60, y,
                             font_size=10,
                             color=WHITE)
            equity_alive = int(round(float(self.players[i].equity_alive) * 100))

            self.viewer.text(f"${self.players[i].stack} (EQ: {equity_alive}%)", x - 60, y + 15, font_size=10,
                             color=WHITE)
            try:
                self.viewer.text(self.players[i].last_action_in_stage, x - 60, y + 30, font_size=10, color=WHITE)
            except IndexError:
                pass
            x_inner = (-face_radius + table_radius - 60) * np.cos(radian) + screen_width / 2
            y_inner = (-face_radius + table_radius - 60) * np.sin(radian) + screen_height / 2
            self.viewer.text(f"${self.player_pots[i]}", x_inner, y_inner, font_size=10, color=WHITE)
            self.viewer.text(f"{self.table_cards}", screen_width / 2 - 40, screen_height / 2, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.community_pot}", screen_width / 2 - 15, screen_height / 2 + 30, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.current_round_pot}", screen_width / 2 - 15, screen_height / 2 + 50,
                             font_size=10,
                             color=WHITE)

            x_button = (-face_radius + table_radius - 20) * np.cos(radian) + screen_width / 2
            y_button = (-face_radius + table_radius - 20) * np.sin(radian) + screen_height / 2
            try:
                if i == self.player_cycle.dealer_idx:
                    self.viewer.circle(x_button, y_button, 5, color=BLUE, thickness=2)
            except AttributeError:
                pass

        self.viewer.update()


# 為了callback到 keras-rl
# todo: Investigate what kind of callback needed here.

class PlayerShell:
    """Player shell"""

    def __init__(self, stack_size, name):
        """Initiaization of an agent"""
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.agent_obj = None
        self.cards = None
        self.num_raises_in_street = {Stage.PREFLOP: 0,
                                     Stage.FLOP: 0,
                                     Stage.TURN: 0,
                                     Stage.RIVER: 0}

    def __repr__(self):
        return f"Player {self.name} at seat {self.seat} with stack of {self.stack} and cards {self.cards}"
