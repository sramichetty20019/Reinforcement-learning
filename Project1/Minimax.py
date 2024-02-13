
import random
import numpy as np

def evaluate_position(grid, player_mark, config):
    score = 0
    for i in range(config.inarow):
        num_windows = count_windows(grid, i + 1, player_mark, config)
        score += (4 ** (i + 1)) * num_windows

    for i in range(config.inarow):
        num_opponent_windows = count_windows(grid, i + 1, get_opponent_mark(player_mark), config)
        score -= (2 ** ((2 * i) + 3)) * num_opponent_windows

    return score

def count_windows(grid, num_discs, player_mark, config):
    num_windows = 0

    # Check horizontally
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_valid_window(window, num_discs, player_mark, config):
                num_windows += 1

    # Check vertically
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_valid_window(window, num_discs, player_mark, config):
                num_windows += 1

    # Check positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_valid_window(window, num_discs, player_mark, config):
                num_windows += 1

    # Check negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_valid_window(window, num_discs, player_mark, config):
                num_windows += 1

    return num_windows

def drop_piece(grid, col, player_mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = player_mark
    return next_grid

def is_valid_window(window, num_discs, player_mark, config):
    return window.count(player_mark) == num_discs and window.count(0) == config.inarow - num_discs

def get_opponent_mark(player_mark):
    return 3 - player_mark

def score_move_a(grid, col, player_mark, config, n_steps=2):
    next_grid = drop_piece(grid, col, player_mark, config)
    valid_moves = [col for col in range(config.columns) if next_grid[0][col] == 0]
    if len(valid_moves) == 0 or n_steps == 0:
        score = evaluate_position(next_grid, player_mark, config)
        return score
    else:
        scores = [score_move_b(next_grid, col, player_mark, config, n_steps-1) for col in valid_moves]
        score = min(scores)
    return score

def score_move_b(grid, col, player_mark, config, n_steps):
    next_grid = drop_piece(grid, col, get_opponent_mark(player_mark), config)
    valid_moves = [col for col in range(config.columns) if next_grid[0][col] == 0]
    if len(valid_moves) == 0 or n_steps == 0:
        score = evaluate_position(next_grid, player_mark, config)
        return score
    else:
        scores = [score_move_a(next_grid, col, player_mark, config, n_steps-1) for col in valid_moves]
        score = max(scores)
    return score

def agent(obs, config):
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    scores = dict(zip(valid_moves, [score_move_a(grid, col, obs.mark, config, 1) for col in valid_moves]))
    best_moves = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return random.choice(best_moves)