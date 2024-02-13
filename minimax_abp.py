
import random
import numpy as np

def evaluate_heuristic(grid, player_mark, config):
    score = 0
    for i in range(config.inarow):
        num_windows = count_windows(grid, i + 1, player_mark, config)
        score += (4 ** (i + 1)) * num_windows

    for i in range(config.inarow):
        num_opponent_windows = count_windows(grid, i + 1, 3 - player_mark, config)
        score -= (2 ** ((2 * i) + 3)) * num_opponent_windows

    return score

def count_windows(grid, num_discs, player_mark, config):
    num_windows = 0

    # Horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_window_valid(window, num_discs, player_mark, config):
                num_windows += 1

    # Vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_window_valid(window, num_discs, player_mark, config):
                num_windows += 1

    # Positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_window_valid(window, num_discs, player_mark, config):
                num_windows += 1

    # Negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_window_valid(window, num_discs, player_mark, config):
                num_windows += 1

    return num_windows

def drop_piece(grid, col, player_mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = player_mark
    return next_grid

def is_window_valid(window, num_discs, player_mark, config):
    return window.count(player_mark) == num_discs and window.count(0) == config.inarow - num_discs

def alpha_beta_minimax(grid, depth, alpha, beta, maximizing_player, player_mark, config):
    valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]

    if depth == 0 or len(valid_moves) == 0:
        return evaluate_heuristic(grid, player_mark, config)

    if maximizing_player:
        max_eval = float('-inf')
        for col in valid_moves:
            next_grid = drop_piece(grid, col, player_mark, config)
            eval = alpha_beta_minimax(next_grid, depth - 1, alpha, beta, False, player_mark, config)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for col in valid_moves:
            next_grid = drop_piece(grid, col, 3 - player_mark, config)
            eval = alpha_beta_minimax(next_grid, depth - 1, alpha, beta, True, player_mark, config)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(grid, player_mark, config, depth=3):
    valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]
    best_score = float('-inf')
    best_move = random.choice(valid_moves)

    for col in valid_moves:
        next_grid = drop_piece(grid, col, player_mark, config)
        score = alpha_beta_minimax(next_grid, depth - 1, float('-inf'), float('inf'), False, player_mark, config)

        if score > best_score:
            best_score = score
            best_move = col

    return best_move

def agent(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    return find_best_move(grid, obs.mark, config)