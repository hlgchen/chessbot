import torch 
from torch import nn
from  torch.optim.lr_scheduler import StepLR

import chess

from bot.model import VFA
from bot.bots import HaiBotLong

import random
import pickle

import torch.utils.tensorboard as tbx
import time

reward_spec = {
    "nothing": -0.1,
    # "capture": 10, 
    "capture_dict": {
        "p": 0.5, 
        "r": 1, 
        "n": 1, 
        "b": 1,
        "q": 2,
    },
    # "enemy_capture": 0,
    "checks": 0.2,
    "checked": -0.2, 
    "fivefold":0,
    "draw": 0, 
    "win": 20, 
    "defeat": -20,
}


def load_data():
    try:
        with open("bin.dat") as f:
            x, y = pickle.load(f)
    except:
        x, y = [], []
    return x, y

def save_data(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def get_reward(last_board, board, last_board_opp, move1, move2, white): 
    outcome = board.outcome()
    reward = reward_spec["nothing"]

    if last_board.is_capture(move1): 
        reward = reward_spec["capture_dict"].get(
            str(last_board.piece_at(move1.to_square)).lower(), 0
        ) 

    if last_board_opp.is_capture(move2):
        reward += reward_spec["capture_dict"].get(
            str(last_board_opp.piece_at(move2.to_square)).lower(), 0
        ) * -1

    if last_board.gives_check(move1): 
        reward += reward_spec["checks"]

    if board.is_check(): 
        reward += reward_spec["checked"]
    
    if outcome is not None: 
        if board.is_fivefold_repetition(): 
            reward += reward_spec["fivefold"]
        elif outcome.winner is None: 
            reward += reward_spec["draw"]
        elif outcome.winner==white: 
            reward = reward_spec["win"]
        else: 
            reward = reward_spec["defeat"]

    return reward
    

def train(run, cont_file=None, cont_file2=None): 
    #hyperparams: 
    C = 20000
    gamma = 0.4

    iter_count = 0
    c_counter = 0
    game_number = 0

    vfa = VFA()
    if cont_file is not None: 
        print(f"evaluation using: {cont_file}")
        vfa.load_state_dict(torch.load(cont_file))
        cont_file = cont_file.split("/")[-1]
        game_number = int(cont_file.split("_")[1].split("m")[-1]) + 1
        iter_count = int(cont_file[:-5].split("_")[-1]) + 1
        print(f"cont. game: {game_number}, iter: {iter_count}")

        if cont_file2 is None: 
            cont_file2 = cont_file


    adam = torch.optim.Adam(vfa.parameters(), lr=0.001)
    scheduler = StepLR(adam, 100, 0.9)
    loss_fn = nn.MSELoss()

    logger = tbx.SummaryWriter(f"out/log/{run}")

    bot1 = HaiBotLong(color="W")
    bot2 = HaiBotLong(color="B")
    bot1.load_vfa_params(vfa.state_dict())
    bot2.load_vfa_params(torch.load(cont_file2))
    print(f"bot1 using: {cont_file}")
    print(f"bot2 using: {cont_file2}")

    replay_buffer = []
    last_five_games = []

    start = time.time()

    for g in range(game_number, game_number + 10000): 

        print(f"chess game: {g}")

        # if old opponent can be beaten regularly (5times in a row) --> new opponent
        last_five_games = last_five_games[-5:]
        if sum(last_five_games) == 5: 
            print("************* new weights OPPONENT *************")
            bot2.load_vfa_params(vfa.state_dict())
            c_counter = 0
        e = 1/(g+3)**0.5

        replay_buffer = replay_buffer[-2000:]


        board = chess.Board()
        move1 = move2 = None
        avg_loss = []
        outcome = None

        while True: 
            
            last_board1 = board.copy()
            action1, features1, checked_feature1 = bot1.play(board, vfa=vfa, e=e, training=True)
            move1 = board.parse_uci(action1)
            board.push(move1)
            outcome = board.outcome()
            
            if move2 is not None: 
            #     # if game ended, add information for bot2 before stopping loop
                if outcome is not None:
                    reward2 = get_reward(last_board2, board, last_board1, move2, move1, white=False)
                    replay_buffer.append((
                            features2, 
                            checked_feature2, 
                            reward2, 
                            board.mirror()
                    ))
                    board.push_uci("0000") # push null move for bot2, so that it's the turn of bot1 again
                    factor = -1 if (abs(reward2) == reward_spec["win"]) else 1
                    replay_buffer.append((features1, checked_feature1, factor * reward2, board))
                    break

            last_board2 = board.copy()
            action2, features2, checked_feature2 = bot2.play(board, e=e, training=True)
            move2 = board.parse_uci(action2)
            board.push(move2)
            outcome = board.outcome()
            
            if move2 is not None: 
                reward1 = get_reward(last_board1, board, last_board2, move1, move2, white=True)
                replay_buffer.append((features1, checked_feature1, reward1, board))

                # if game ended, add information for bot2 before stopping loop
                if outcome is not None:
                    factor = -1 if (abs(reward1) == reward_spec["win"]) else 1
                    board.push_uci("0000") # push null move for bot1, so that it's the turn of bot2 again
                    replay_buffer.append((
                        features2, 
                        checked_feature2, 
                        factor * reward1, 
                        board.mirror()
                    ))
                    break
            

            # minibatch training
            targets = []
            features = []
            checked_feature = []
            minibatch = random.choices(replay_buffer, k=58) + replay_buffer[-8:]
            for tup in minibatch: 
                q_vfa_old = bot1.get_value_best_move(tup[3], vfa)
                if q_vfa_old is None: 
                    y = torch.Tensor([[tup[2]]])
                else: 
                    y = torch.Tensor([tup[2]]) + gamma *  q_vfa_old

                targets.append(y)
                features.append(tup[0])
                checked_feature.append(tup[1])

            targets = torch.cat(targets)
            features = torch.cat(features, dim=0)
            checked_feature = torch.cat(checked_feature, dim=0)
            preds = vfa(features, checked_feature)

            adam.zero_grad()
            loss = loss_fn(preds, targets)
            loss.backward()
            adam.step()

            # ****** progress monitoring ******
            iter_count += 1
            c_counter += 1

            if iter_count % 20 == 0: 
                end = time.time()
                diff = round(end-start, 6)
                print(f"iter: {iter_count}, loss: {loss}, took {diff} seconds")
                start = time.time()
            ## after certain number of iteration: update vfa params for evaluation
            if c_counter % C == 0: 
                print("*********** new weights EVALUATION ***********")
                bot1.load_vfa_params(vfa.state_dict())
            
            avg_loss.append(loss)

        print(f"Game {g} outcome: {outcome}")
        last_five_games.append(outcome.winner ==1)

        if g % 10 == 0: 
            save_data(board.move_stack, f"out/games/{run}/game_{g}.dat")
            torch.save(vfa.state_dict(), f"out/models/{run}/m_{g}_iter_{iter_count}.ckpt")
            
        logger.add_scalar("loss", sum(avg_loss)/len(avg_loss), g)
        logger.add_scalar("number_moves", len(board.move_stack),g)
        logger.add_scalar("number_living", len(board.piece_map()),g)
        logger.flush()

        scheduler.step()



if __name__ == "__main__": 
    train(
        run="local2",
        cont_file="out/models/champs/m_410_iter_35396.ckpt", 
        cont_file2="out/models/champs/m_410_iter_35396.ckpt", 
        )