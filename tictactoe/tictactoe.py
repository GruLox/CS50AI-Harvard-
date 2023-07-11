"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    
    if terminal(board):
        return None

    xCount = 0
    oCount = 0
    for row in board:
        for cell in row:
            if (cell == O):
                oCount += 1
            elif (cell == X):
                xCount += 1
    
    return X if xCount == oCount else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    if terminal(board):
        return None

    possibleActions = []
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == EMPTY:
                possibleActions.append((i, j))
    
    return possibleActions


def result(board: list, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if action not in actions(board):
        raise ValueError
    
    boardCopy = deepcopy(board)
    playerWithTurn = player(board)

    boardCopy[action[0]][action[1]] = playerWithTurn

    return boardCopy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        elif board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]
    
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[2][0] == board[1][1] == board[0][2]:
        return board[2][0]
    
    return None
    
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    isEmptyCellOnBoard = any(cell is None for row in board for cell in row)

    if winner(board) or not isEmptyCellOnBoard:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winnerPlayer = winner(board)
    if not winnerPlayer:
        return 0
    elif winnerPlayer == O:
        return -1
    else:
        return 1

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    currentPlayer = player(board)

    actionValues = []

    if (currentPlayer == X):
        for action in actions(board):
            v = minValue(result(board, action))
            actionValues.append(v)
        
        optimalActionIndex = actionValues.index(max(actionValues))
        return actions(board)[optimalActionIndex]
            
    else:
        for action in actions(board):
            v = maxValue(result(board, action))
            actionValues.append(v)
        
        optimalActionIndex = actionValues.index(min(actionValues))
        return actions(board)[optimalActionIndex]


def maxValue(board):
    v = float('-inf')

    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        v = max(v, minValue(result(board, action)))
    return v


def minValue(board):
    v = float('inf')

    if (terminal(board)):
        return utility(board)
    
    for action in actions(board):
        v = min(v, maxValue(result(board, action)))
    return v