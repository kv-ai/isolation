"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

import operator

infinity = float('inf')
num_nodes = 0
MAX = float("inf")
MIN = float("-inf")
NO_MOVES_LEFT = (-1,-1)

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Whoever is isolated loses the game.
    ##Improve the heuristic score by subtracting AI agents moves and
    ##Opponent's, 'super aggresive improved score'
    if game.is_loser(player):
        return MIN
 
    if game.is_winner(player):
        return MAX

    my_score = 0;

    opponent_score = 0;
    
    number_of_my_moves = len(game.get_legal_moves(player))
    number_of_opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))    
    
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
    if number_of_my_moves != number_of_opponent_moves:
        return float(number_of_my_moves - number_of_opponent_moves)
    else:    
        for legal_move in my_moves:
            if legal_move in opponent_moves:
                my_score = my_score + 10
            else:
                my_score = my_score - 1
                
        for legal_move in opponent_moves:
            if legal_move in my_moves:
                opponent_score = opponent_score - 10
            else:
                opponent_score = opponent_score + 1
        return float(my_score - opponent_score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Whoever is isolated loses the game.
    ##Subtract the open move score and number of next possible moves 
    ##of the opponent, 'improved score (aggresive)'    
    if game.is_loser(player):
        return MIN

    if game.is_winner(player):
        return MAX

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - (3*opponent_moves))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    ##Whoever is isolated loses the game.
    ##Count all the next possible moves each player has at
    ##any given time, possible moves means less chance of 
    ##being isolated, "open move score"
    if game.is_loser(player):
        return MIN

    if game.is_winner(player):
        return MAX
    
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if my_moves != opponent_moves:
        return float(my_moves - opponent_moves)
    else:
        #calculate manhattan distance to the center of the board.
        #calculate the center coordinates of the game board.
        center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)
        #get agents coordinates
        my_y_pos, my_x_pos = game.get_player_location(player)
        #get opponents coordinates
        opponent_y_pos, opponent_x_pos = game.get_player_location(game.get_opponent(player))
        my_distance = abs(my_y_pos - center_y_pos) + abs(my_x_pos - center_x_pos)
        opponent_distance = abs(opponent_y_pos - center_y_pos) + abs(opponent_x_pos - center_x_pos)
        return float((opponent_distance - my_distance) / 10.0) 


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = NO_MOVES_LEFT
        moves = game.get_legal_moves(self)
        if len(moves) > 0:
            best_move = moves[0]
        try:
            ##SearchTimeout exception is raised in calculating
            ##min_value and max_value which will automatically be 
            ##be raised, when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move

        # Return the best move
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def terminal_state(game, self):
            if len(game.get_legal_moves(self)) == 0:
                return True # game over!
            else:
                return False

        def max_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or terminal_state(game, self):
                return self.score(game, self)
            v = MIN
            for a in game.get_legal_moves(self):
                v = max(v, min_value(self, game.forecast_move(a), depth-1))
            return v

        def min_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or terminal_state(game, game.active_player):
                return self.score(game, self)
            v = MAX
            for a in game.get_legal_moves(game.active_player):
                v = min(v, max_value(self, game.forecast_move(a), depth-1))
            return v


        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if terminal_state(game, self):
            return NO_MOVES_LEFT


        best_move = NO_MOVES_LEFT
        moves = game.get_legal_moves(self)
        if len(moves) > 0:
            best_move = moves[0]

        best_move = max(game.get_legal_moves(self),
                        key=lambda a: min_value(self, game.forecast_move(a), depth-1))

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Start with ID depth 1 and increase it later
        depth = 1
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = NO_MOVES_LEFT
        moves = game.get_legal_moves(self)
        if len(moves) > 0:
            best_move = moves[0]


        while True:
            try:
                ##SearchTimeout exception is raised in calculating
                ##min_value and max_value which will automatically be 
                ##be raised, when the timer is about to expire.
                best_move = self.alphabeta(game, depth)
                depth = depth + 1
            except SearchTimeout:
                return best_move

        # Return the best move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def terminal_state(game, self):
            if len(game.get_legal_moves(self)) == 0:
                ## final move, end of game
                return True 
            else:
                return False

        def max_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or terminal_state(game, self):
                return self.score(game, self)
            v = MIN
            for a in game.get_legal_moves(self):
                v = max(v, min_value(self, game.forecast_move(a), depth-1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or terminal_state(game, game.active_player):
                return self.score(game, self)
            v = MAX
            for a in game.get_legal_moves(game.active_player):
                v = min(v, max_value(self, game.forecast_move(a), depth-1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if terminal_state(game, self):
            return NO_MOVES_LEFT

        beta = MAX
        best_score = MIN
        best_move = NO_MOVES_LEFT
        moves = game.get_legal_moves(self)
        if len(moves) > 0:
            best_move = moves[0]

        v = MIN
        for a in game.get_legal_moves(self):
            v = max(v, min_value(self, game.forecast_move(a), depth-1, alpha, beta))
            if v > best_score:
                best_move = a
                best_score = v
            alpha = max(alpha, v)
            
        return best_move
