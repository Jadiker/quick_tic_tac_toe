# used to save and load what the AI has learned
import pickle
import os
import random
from queue import Empty
from stat import FILE_ATTRIBUTE_COMPRESSED

SAVE_FILE = "knowledge.pkl"
HUMAN_OPPONENT = "human opponent"
AI_OPPONENT = "ai opponent"
AI_PLAYER = "ai"
OPPONENT_PLAYER = "opp"
NOT_OVER = "not over"
TIE = "tie"
EMPTY = 0
HUMAN_GAME_FACTOR = 5/6
REAL_AI_GAME_FACTOR = 2/3
FAKE_GAME_FACTOR = 1/10
WIN_SCORE = 5
LOSE_SCORE = -100
TIE_SCORE = 1
FAKE_GAME_AMOUNT = 3
DEFAULT_SCORE = 0
# how much less to update the score of a move that happened far away from the end
LEVEL_FACTOR = 1/2
EXPLOIT_CHANCE = 0.9

# check if the save file exists
if os.path.isfile(SAVE_FILE):
    # load the save file
    with open(SAVE_FILE, "rb") as save_file:
        move_scores = pickle.load(save_file)
else:
    # create the data structure that we'll save later
    # format: {scenario: {move: score, ...}, ...}
    # move_scores[scenario][move1] gives the score for making that move in that game
    move_scores = {}

# a scenario is an unmodifiable game (turn the list into a tuple)
# this is done because lists can't be keys in dictionaries, but tuples can be
def game_to_scenario(game):
    return tuple(game)

def update_from_data(data, update_factor, final_score):
    # data is in the format [[scenario, move]]
    # It's a sequence of moves that were made in the game
    # The further a move is from the end of the game, the less we should update it
    number_of_moves = len(data)
    # make a list that increases by one for each move, starting at 1
    increasing = []
    for i in range(number_of_moves):
        increasing.append(i + 1)
    # we want the earlier moves to be discounted the most
    # so we actually want the list to be decreasing
    # so, we should reverse it
    increasing.reverse()

    # use the increasing list to come up with the level factors
    level_factors = []
    for increase in increasing:
        level_factors.append(LEVEL_FACTOR ** increase)
    
    # multiply the level factors by the update factor
    factors = []
    for level_factor in level_factors:
        factors.append(level_factor * update_factor)
    
    datapoint_index = 0
    for datapoint in data:
        scenario = datapoint[0]
        move = datapoint[1]
        factor = factors[datapoint_index]
        update_scenario_score_with_factor(scenario, move, final_score, factor)
        datapoint_index = datapoint_index + 1

def update_scenario_score_with_factor(scenario, move, score, factor):
    if scenario not in move_scores.keys():
        move_scores[scenario] = {}
    if move not in move_scores[scenario].keys():
        move_scores[scenario][move] = DEFAULT_SCORE

    previous_score = move_scores[scenario][move]
    move_scores[scenario][move] = previous_score + (score - previous_score) * factor

def make_move(game, space, player):
    game[space] = player
    return game

def copy_list(given_list):
    new_list = []
    for thing in given_list:
        new_list.append(thing)
    return new_list

def get_possible_moves(game):
    # get the indexes of the empty spaces in the board
    possible_moves = []
    for i in range(len(game)):
        if game[i] == EMPTY:
            possible_moves.append(i)
    return possible_moves

def check_winner(game):
    for player in [AI_PLAYER, OPPONENT_PLAYER]:
        # check rows
        if game[0] == player and game[1] == player and game[2] == player:
            return player
        if game[3] == player and game[4] == player and game[5] == player:
            return player
        if game[6] == player and game[7] == player and game[8] == player:
            return player
        # check columns
        if game[0] == player and game[3] == player and game[6] == player:
            return player
        if game[1] == player and game[4] == player and game[7] == player:
            return player
        if game[2] == player and game[5] == player and game[8] == player:
            return player
        # check diagonals
        if game[0] == player and game[4] == player and game[8] == player:
            return player
        if game[2] == player and game[4] == player and game[6] == player:
            return player
    possible_moves = get_possible_moves(game)
    # determine if it's a tie or if the game is still in progress
    if len(possible_moves) == 0:
        return TIE
    else:
        return NOT_OVER

def player_to_string(player):
    if player == AI_PLAYER:
        return "X"
    elif player == OPPONENT_PLAYER:
        return "O"
    else:
        return "-"

def display_game(game):
    # turn the player strings like "ai" and "opp" into displayable X's and O's
    strings = []
    for i in range(len(game)):
        strings.append(player_to_string(game[i]))
    # print the board
    print(strings[0] + strings[1] + strings[2])
    print(strings[3] + strings[4] + strings[5])
    print(strings[6] + strings[7] + strings[8])
    

def train_ai(opponent):
    # game of Tic Tac Toe with all nine slots empty
    game = [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    who_goes_first = random.choice([OPPONENT_PLAYER, AI_PLAYER])
    is_game_over = False
    whose_turn = who_goes_first
    # format: [[scenario, move], ...]
    game_data = []
    # there's no winner yet
    winner = None

    while is_game_over is False:
        print("Current game state:")
        display_game(game)
        if whose_turn == AI_PLAYER:
            # try each move and learn from them
            learn_from_simulation(game, opponent)

            # randomly decide whether to explore or exploit
            random_explore_number = random.random()
            if random_explore_number > EXPLOIT_CHANCE:
                # explore
                # pick a move at random
                possible_moves = get_possible_moves(game)
                move = random.choice(possible_moves)
            else:
                # exploit
                # pick the move that has the highest score
                highest_score = float('-inf')
                move = None
                possible_moves = get_possible_moves(game)
                for possible_move in possible_moves:
                    # these will all exist because it will have tried all of these moves during the simulation
                    # (that is, no need to check if they're empty / nonexistent)
                    scenario = game_to_scenario(game)
                    score = move_scores[scenario][possible_move]
                    if score > highest_score:
                        # this is the highest scoring move so far - save it!
                        move = possible_move
                        highest_score = score
            
            # save the game as a scenario before it gets modified by the move
            scenario = game_to_scenario(game)
            game_data.append([scenario, move])
            game = make_move(game, move, AI_PLAYER)
            print(f"The AI played move {move}.")
            whose_turn = OPPONENT_PLAYER
        else:
            # opponent's turn
            if opponent == HUMAN_OPPONENT:
                # get the move from the user
                move = get_move_from_user(game)
            else:
                # get a random move
                possible_moves = get_possible_moves(game)
                move = random.choice(possible_moves)
            game = make_move(game, move, OPPONENT_PLAYER)
            whose_turn = AI_PLAYER
        
        # determine if the game is over
        winner = check_winner(game)
        if winner is not NOT_OVER:
            is_game_over = True

    if winner == TIE:
        print("Tie game!")
    else:
        print(f"The winner is {player_to_string(winner)}!")
    display_game(game)

    # tell the AI how well it did
    if winner == AI_PLAYER:
        score = WIN_SCORE
    elif winner == OPPONENT_PLAYER:
        score = LOSE_SCORE
    else:
        score = TIE_SCORE

    # set how much we want the results to influence what the AI learns
    if opponent == AI_PLAYER:
        update_factor = REAL_AI_GAME_FACTOR
    else:
        # opponent == OPPONENT_PLAYER:
        update_factor = HUMAN_GAME_FACTOR

    # update the scores of the moves that were played
    update_from_data(game_data, update_factor, score)


def get_move_from_user(game):
    # get the move from the user
    print("Enter a number between 0 and 8:")
    move = int(input())
    while move not in get_possible_moves(game):
        print("Invalid move. Try again:")
        move = int(input())
    return move

def learn_from_simulation(game, opponent):
    # try each possible move a certain number of times and
    # see how well that move (plus other random moves) does against a random opponent
    # Update the scores accordingly
    # there will always be moves available because the game is not over
    possible_moves = get_possible_moves(game)
    scenario = copy_list(game)
    for move in possible_moves:
        game_data = []
        # create a copy of the game that the simulations can happen on
        simulation_game = copy_list(game)
        scenario = game_to_scenario(simulation_game)
        game_data.append([scenario, move])
        simulation_game = make_move(simulation_game, move, AI_PLAYER)
        for i in range(FAKE_GAME_AMOUNT):
            copied_simulation_game = copy_list(simulation_game)
            copied_game_data = copy_list(game_data)
            play_random_and_update(copied_simulation_game, copied_game_data)

def play_random_and_update(copied_simulation_game, copied_game_data):
    whose_turn = OPPONENT_PLAYER
    while check_winner(copied_simulation_game) == NOT_OVER:
        # make a random move for the current player
        possible_moves = get_possible_moves(copied_simulation_game)
        move = random.choice(possible_moves)
        # turn the game into a scenario before the move is made so we can save the scenario if needed later
        scenario = game_to_scenario(copied_simulation_game)
        copied_simulation_game = make_move(copied_simulation_game, move, whose_turn)
        if whose_turn == AI_PLAYER:
            copied_game_data.append([scenario, move])
            whose_turn = OPPONENT_PLAYER
        else:
            whose_turn = AI_PLAYER
    
    # set the score based on the outcome
    winner = check_winner(copied_simulation_game)
    if winner == AI_PLAYER:
        score = WIN_SCORE
    elif winner == OPPONENT_PLAYER:
        score = LOSE_SCORE
    else:
        score = TIE_SCORE
    # these are fake random games the AI is playing, so we use the appropriate scaling
    update_from_data(copied_game_data, FAKE_GAME_FACTOR, score)


train_ai(HUMAN_OPPONENT)

# save the training data

with open(SAVE_FILE, "wb") as save_file:
    pickle.dump(move_scores, save_file)