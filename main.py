from connect4.mcts_ia import train_mcts_once, train_mcts_during
from connect4.connect4 import *
from collections import defaultdict
import numpy as np


def utils_print(grid):
    print_grid = grid.astype(str)
    print_grid[print_grid == '-1'] = 'X'
    print_grid[print_grid == '1'] = 'O'
    print_grid[print_grid == '0'] = ' '
    res = str(print_grid).replace("'", "")
    res = res.replace('[[', '[')
    res = res.replace(']]', ']')
    print(' ' + res)
    print('  ' + ' '.join('0123456'))

if __name__ == '__main__':

    mcts = None

    for i in range(100):
        mcts = train_mcts_once(mcts)

    print('training finished')
    while True:
        # test AI with real play
        grid = create_grid()
        round = 0
        training_time = 2000
        node = mcts
        utils_print(grid)
        while True:
            if (round % 2) == 0:
                move = int(input())
                new_node = node.get_children_with_move(move)
                node = train_mcts_during(node, training_time).get_children_with_move(move)
            else:
                new_node, move = node.select_move()
                node = train_mcts_during(node, training_time)
                # print([(n.win, n.games) for n in node.children])
                node, move = node.select_move()

            grid, winner = play(grid, move)

            utils_print(grid)


            assert np.sum(node.state - grid) == 0, node.state
            if winner != 0:
                print('Winner : ', 'X' if winner == -1 else 'O')
                break
            round += 1


    from pdb import set_trace; set_trace()

