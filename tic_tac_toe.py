'''
This is a reinforcement learning AI that learns to play the game of TicTacToe by
...playing against random opponents - and you!

I've coded so that it doesn't require understanding any advanced coding techniques.
(I don't use classes, list comprehensions, etc.)

In this code, a TicTacToe game is represented by a list of 9 values.
For example, the board

OXO
-OX
XOX

is represented in the code with
["opp", "ai", "opp", "empty", "opp", "ai", "ai", "opp", "ai"]

I go from right-to-left, top-to-bottom (just like reading).
("opp" is short for "opponent")

Then, the AI stores a dictionary (database) of all the game boards its ever seen.
Each game board has its own dictionary (database) of moves that the AI has made in that scenario.
Then, each move is given a score of how good it is for the AI.

All moves start out with a DEFAULT_SCORE of 0.
If a move is definitely a win, the score should be closer to a WIN_SCORE of 5.
If it's definitely a loss, it should be closer to a LOSE_SCORE of -100.
If the move definitely leads to a tie, it should be closer to a TIE_SCORE of 1.
(Feel free to play around with these numbers!)

When the AI needs to make a move, it tries all of the moves!
It plays 3 (FAKE_GAME_AMOUNT) completely random games for each of the moves it tries.
Then, it updates the move scores based on how those games turned out.

Move scores are updated using the following rule:
NEW_MOVE_SCORE = (GAME_SCORE - OLD_MOVE_SCORE) *
                 GAME_FACTOR *
                 ((how many moves this move was from the last move of the game) * DISCOUNT_FACTOR)
(The GAME_SCORE is the outcome of the game: either WIN_SCORE, TIE_SCORE, or LOSE_SCORE.)
(GAME_FACTOR is set based on who the AI is playing. It will be higher
..when playing against humans so that it updates scores more from those games.)
(DISCOUNT_FACTOR makes it so that moves early on in the game aren't updated as much
...because we're not actually certain that they lead to that outcome.
For example, if your first move is in the center, but you lose the game,
...that doesn't mean that the first move was actually bad.)

This rule is known as a Bellman Update equation, and is the main equation used in reinforcement learning!

Basically, you take the old score and try to move it closer to the score you got for the game.
But since you want to learn less from playing bad opponents, and you also aren't certain that you couldn't have played better,
...you don't want to set the score for the move to just be the outcome of the game.
So, you shrink how much you change the score based on who you're playing
...and how many moves this move was from the end of the game.
Moves closer to the end of the game were probably more responsible for losing the game, so they get updated more.
Learning from better opponents is probably better, so you update moves more when playing against good opponents.

Then, after playing these random games and updating the scores, it picks a move.
It has a small chance to pick a non-optimal move so that it learns more about other scenarios.
If it picks an optimal move, we say it's "exploiting", and if it picks a non-optimal move, we say it's "exploring".

Finally, when the game is over, it saves what it learned so it can use that info for the next game!

With that said, here's the code! Enjoy! :)
'''

# for saving and loading what the AI has learned
import pickle
# for checking to make sure files exist
import os
# for generating random numbers
import random

# where the AI's knowledge is stored
SAVE_FILE = "knowledge.pkl"
# should the AI always make what it thinks is the best move?
# True if it should make the best move
# False if you want it to explore and learn more
ALWAYS_EXPLOIT = False
# if it's not always exploiting, this is the chance it will not explore (a move that it doesn't think is the best)
EXPLOIT_CHANCE = 0.9
# before making a move, the AI tries each move and sees how that move does against a random player
# how many games should it play against that random player in order to figure out what move is the best?
# more games will make it seem more intelligent, but take longer to move
FAKE_GAME_AMOUNT = 3
# how much to weight games that are played against humans
HUMAN_GAME_FACTOR = 5/6
# how much to weight games that are played against the AI
REAL_AI_GAME_FACTOR = 2/3
# how much to weight games that are played against the random AI when testing moves
FAKE_GAME_FACTOR = 1/10
# how much less to update the score of a move that happened early on in a game
# this will be multiplied per move from the last move
# for example, if this is 1/2, then the first move of a 3-move game (the shortest possible game) would only be updated by 1/8 the amount as the last move
# (note that the score of a move is how good the AI thinks the move is)
DISCOUNT_FACTOR = 1/2

# what should the score of a move that leads to a win be?
WIN_SCORE = 5
# what should the score of a move that leads to a loss be?
LOSE_SCORE = -100
# what should the score of a move that leads to a tie be?
TIE_SCORE = 1
# What should the score of a move that we have no data on be?
DEFAULT_SCORE = 0

# constants that I use in the code so I don't accidentally mistype the things they refer to
HUMAN_OPPONENT = "human opponent"
AI_OPPONENT = "ai opponent"
AI_PLAYER = "ai"
OPPONENT_PLAYER = "opp"
NOT_OVER = "not over"
TIE = "tie"
EMPTY = "empty"


# create the data structure that we'll save later
# format: {scenario: {move: score, ...}, ...}
# move_scores[scenario][move1] gives the score for making that move in that scenario
move_scores = {}

# check if the save file exists
if os.path.isfile(SAVE_FILE):
    # replace the empty data with the saved data
    with open(SAVE_FILE, "rb") as save_file: # open the file
        move_scores = pickle.load(save_file) # load the data into the variable

# ******* See the bottom of the file to understand what's actually being run ********

# a scenario is an unmodifiable game (turn the modifiable list into an unmodifiable tuple)
# (this is done because lists can't be keys in dictionaries, but tuples can be)
def game_to_scenario(game):
    return tuple(game)

# this is the heart of the AI
# given data about what moves were made in what scenarios and what score those moves lead to, update the AI to make better choices
def update_from_data(data, update_factor, final_score):
    # data is in the format [[scenario, move], ...]
    # It's a sequence of moves that were made in the game
    # update_factor and final_score are numbers

    # the further a move is from the end of the game, the less we should update it
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
        level_factors.append(DISCOUNT_FACTOR ** increase)
    
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
            if ALWAYS_EXPLOIT == True:
                random_explore_number = 0
            else:
                random_explore_number = random.random()
            if random_explore_number > EXPLOIT_CHANCE:
                # explore
                # pick a move at random
                print("Exploring to see what happens...")
                possible_moves = get_possible_moves(game)
                move = random.choice(possible_moves)
            else:
                # exploit
                # pick the move that has the highest score
                print("Choosing the best move...")
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
            print(f"The AI played move {move + 1}.")
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
    print("Enter a number between 1 and 9:")
    # convert human numbers (starting at 1) into indexes (starting at 0)
    move = int(input()) - 1
    while move not in get_possible_moves(game):
        print("Invalid move. Try again:")
        move = int(input()) - 1
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

# *******The code is actually run here********
# you can also have it play against an AI_OPPONENT
train_ai(HUMAN_OPPONENT)

# save the training data
with open(SAVE_FILE, "wb") as save_file:
    pickle.dump(move_scores, save_file)