import numpy as np
import random

from .mcts import Node
from .connect4 import *


def random_play(grid):
    """
    Play a random game starting by state and player
    Return winner
    """

    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        selected_move = random.choice(moves)
        player_to_play = get_player_to_play(grid)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play

def random_play_improved(grid):

    def get_winning_moves(grid, moves, player):
        return [move for move in moves if play(grid, move, player=player)[1]]

    # If can win, win
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        player_to_play = get_player_to_play(grid)

        winning_moves = get_winning_moves(grid, moves, player_to_play)
        loosing_moves = get_winning_moves(grid, moves, -player_to_play)

        if len(winning_moves) > 0:
            selected_move = winning_moves[0]
        elif len(loosing_moves) == 1:
            selected_move = loosing_moves[0]
        else:
            selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play


def train_mcts_during(mcts, training_time):
    import time
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts)
        current = int(round(time.time() * 1000))
    return mcts

def train_mcts_once(mcts=None):

    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)

    node = mcts

    # selection
    while node.children is not None:
        # Select highest uct
        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # expansion, no expansion if terminal node
    moves = valid_move(node.state)
    if len(moves) > 0:

        if node.winner == 0:

            states = [(play(node.state, move), move) for move in moves]
            node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # simulation
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner

        # backpropagation
        parent = node
        while parent is not None:
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
            parent = parent.parent


    else:
        print('no valid moves, expended all')

    return mcts

