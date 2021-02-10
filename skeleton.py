import gym
import random
import requests
import numpy as np
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

# SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["fi6470al-s"]


def call_server(move):
    res = requests.post(SERVER_ADRESS + "move",
                        data={
                            "stil_id": STIL_ID,
                            "move": move,
                            # -1 signals the system to start a new game. any running game is counted as a loss
                            "api_key": API_KEY,
                        })
    # For safety some respose checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()['status']:
        print("Server returned a bad status. Return message: ")
        print(res.json()['msg'])
        exit()
    return res


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env):
    env.change_player()  # change to opponent
    avmoves = env.available_moves()
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarrantee a win
    action = random.choice(list(avmoves))

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


def student_move(state):
    """
    WORKS
    Selects the best move the minimax algorithm found.
    """
    return minimax(state, 7)


def minimax(state, depth):
    """
    WORKS
    Minimax algorithm with Alpha-beta pruning to find score of a state
    """
    values = {x: None for x in range(7)}

    for child in children_of_state(state, 1):
        v = min_value(child, depth - 1, -np.inf, np.inf)
        values[move_made(state, child)] = v

    for key, value in dict(values).items():
        if value is None:
            del values[key]

    if not values:
        random_move = random.choice(children_of_state(state, 1))
        move = move_made(state, random_move)
    else:
        move = max(values, key=lambda k: values[k])

    return move


def min_value(state, depth, alpha, beta):
    """
    WORKS
    Finds the minimum min-maxed value of a game-state
    """
    if player_win(state, -1):
        return -np.inf

    if depth == 0:
        return eval_move(state)

    v = np.inf

    for child in children_of_state(state, -1):
        new_value = max_value(child, depth - 1, alpha, beta)
        v = min(v, new_value)
        beta = min(beta, new_value)
        if beta <= alpha:
            break

    return v


def max_value(state, depth, alpha, beta):
    """
    WORKS
    Finds the maximum min-maxed value of a game-state
    """
    if player_win(state, 1):
        return np.inf

    if depth == 0:
        return eval_move(state)

    v = -np.inf

    for child in children_of_state(state, 1):
        new_value = min_value(child, depth - 1, alpha, beta)
        v = max(v, new_value)
        alpha = max(alpha, new_value)
        if beta <= alpha:
            break

    return v


def player_win(state, player):
    """
    WORKS
    Checks if player has won the game given its state
    """

    """
    Checks for wins in rows
    """
    for i in range(len(state)):
        for j in range(len(state[i])-3):
            if sum(state[i][j:j+4]) == 4*player:
                return True

    """
    Checks for wins in columns
    """
    state_transpose = np.transpose(state)
    for i in range(len(state_transpose)):
        for j in range(len(state_transpose[i])-3):
            if sum(state_transpose[i][j:j+4]) == 4*player:
                return True

    """
    Checks for wins in right-going diagonals
    """
    for i in range(len(state)-3):
        for j in range(len(state[i])-3):
            if state[i][j]+state[i+1][j+1]+state[i+2][j+2]+state[i+3][j+3] == 4*player:
                return True

    """
    Checks for wins in left-going diagonals
    """
    xs = [x+3 for x in range(len(state)-3)]
    for i in xs:
        for j in range(len(state[i])-3):
            if state[i][j]+state[i-1][j+1]+state[i-2][j+2]+state[i-3][j+3] == 4*player:
                return True

    return False


def eval_move(state):
    """
    WORKS
    Evaluates a score for a board state
    Returns <0 if player 1 is likely to win
    Returns 0 if equal
    returns >0 if player -1 is likely to win
    """
    eval_weights = [[3, 4, 5, 7, 5, 4, 3],
                    [4, 6, 8, 10, 8, 6, 4],
                    [5, 8, 11, 13, 11, 8, 5],
                    [5, 8, 11, 13, 11, 8, 5],
                    [4, 6, 8, 10, 8, 6, 4],
                    [3, 4, 5, 7, 5, 4, 3]]

    utility = 0
    for i in range(len(eval_weights)):
        for j in range(len(eval_weights[i])):
            if state[i][j] == 1:
                utility += eval_weights[i][j]
            elif state[i][j] == -1:
                utility -= eval_weights[i][j]

    return utility


def move_made(state, child):
    """
    WORKS
    Finds what move has been made given a state and a child of that state
    """
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] == 0) and (child[i][j] != 0):
                return j


def children_of_state(state, player):
    """
    WORKS
    Finds the children of a board state, given player 1 or -1.
    """

    children = []

    moves = [3, 4, 2, 5, 1, 6, 0]

    for i in range(len(state)):
        for move in moves:
            if state[i][move] == 0 and i == 5:
                child = state.copy()
                child[i][move] = player
                children.append(child)

            elif state[i][move] == 0 and state[i+1][move] != 0:
                child = state.copy()
                child[i][move] = player
                children.append(child)

    return children


def play_game(vs_server=False):
    """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

    # default state
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(-1)  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()['msg'])
        botmove = res.json()['botmove']
        state = np.array(res.json()['state'])
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print('You start!')
            print()
        else:
            print('Bot starts!')
            print()

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:
        # Select your move
        stmove = student_move(state)  # TODO: change input here

        # make both student and bot/server moves
        if vs_server:
            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()['msg'])

            # Extract response values
            result = res.json()['result']
            botmove = res.json()['botmove']
            state = np.array(res.json()['state'])
        else:
            if student_gets_move:
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tied to make an illegal move! Games ends.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
            elif result == 0.5:
                print("It's a draw!")
            elif result == -1:
                print("You lost!")
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current gamestate
        print(state)
        print()


def main():
    play_game(vs_server=True)


if __name__ == "__main__":
    main()
