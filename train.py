import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import chess

from bot.model import VFA
from bot.bots import HaiBotLong

import random
import pickle

import torch.utils.tensorboard as tbx
import time

reward_spec = {
    "nothing": -1,
    "capture_dict": {
        "p": 5,
        "r": 25,
        "n": 15,
        "b": 15,
        "q": 45,
    },
    # "enemy_capture": 0,
    "checks": 2,
    "checked": -2,
    "fivefold": 0,
    "draw": 0,
    "win": 200,
    "defeat": -200,
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
        reward += (
            reward_spec["capture_dict"].get(
                str(last_board_opp.piece_at(move2.to_square)).lower(), 0
            )
            * -1
        )

    if last_board.gives_check(move1):
        reward += reward_spec["checks"]

    if board.is_check():
        reward += reward_spec["checked"]

    if outcome is not None:
        if board.is_fivefold_repetition():
            reward += reward_spec["fivefold"]
        elif outcome.winner is None:
            reward += reward_spec["draw"]
        elif outcome.winner == white:
            reward = reward_spec["win"]
        else:
            reward = reward_spec["defeat"]

    return reward


def continue_training_from_file(file):
    cont_file = file.split("/")[-1]
    game_number = int(cont_file.split("_")[1].split("m")[-1]) + 1
    iter_count = int(cont_file[:-5].split("_")[-1]) + 1
    print(f"cont. game: {game_number}, iter: {iter_count}")
    return game_number, iter_count


def train(run, action_vfa_file=None, evaluation_vfa_file=None, opponent_vfa_file=None):
    # hyperparams:
    C = 20000
    gamma = 0.5

    iter_count = 0
    c_counter = 0
    game_number = 0
    learning_rate = 0.001

    action_vfa = VFA()
    evaluation_vfa = VFA()
    opponent_vfa = VFA()

    if action_vfa_file is not None:
        print(f"action selection using: {action_vfa_file}")
        action_vfa.load_state_dict(torch.load(action_vfa_file), strict=False)
        game_number, iter_count = continue_training_from_file(action_vfa_file)
    if evaluation_vfa_file is not None:
        print(f"action evaluation using: {evaluation_vfa_file}")
        evaluation_vfa.load_state_dict(torch.load(evaluation_vfa_file), strict=False)
    if opponent_vfa_file is not None:
        print(f"opponent using: {opponent_vfa_file}")
        opponent_vfa.load_state_dict(torch.load(opponent_vfa_file), strict=False)

    adam = torch.optim.Adam(action_vfa.parameters(), lr=learning_rate)
    scheduler = StepLR(adam, 100, 0.5)
    loss_fn = nn.MSELoss()

    logger = tbx.SummaryWriter(f"out/log/{run}")

    bot1 = HaiBotLong()
    bot2 = HaiBotLong()
    bot2.load_new_vfa(opponent_vfa)

    replay_buffer = []
    last_five_games = []

    start = time.time()

    for g in range(game_number, game_number + 10000):

        train_white = random.randint(0, 1) == 0
        bot1.w = train_white
        bot2.w = not train_white
        print(f"chess game: {g}, train_white: {train_white}")

        # if old opponent can be beaten regularly (5times in a row) --> new opponent
        last_five_games = last_five_games[-5:]
        if sum(last_five_games) == 5:
            print("************* new weights OPPONENT *************")
            bot2.load_new_vfa(action_vfa)
            print("*********** new weights EVALUATION ***********")
            evaluation_vfa.load_state_dict(action_vfa.state_dict())
            for param in evaluation_vfa.parameters():
                param.requires_grad = False
            game_number = g
            c_counter = 0
            last_five_games = []
            for ad in adam.param_groups:
                ad["lr"] = learning_rate
        e = 1 / (g - game_number + 40) ** 0.7

        replay_buffer = replay_buffer[-4000:]

        board = chess.Board()
        move1 = move2 = outcome = None
        avg_loss = []

        while True:

            if train_white:
                last_board1 = board.copy()
                action1, features1, checked_feature1 = bot1.play(
                    board, vfa=action_vfa, e=e, training=True
                )
                move1 = board.parse_uci(action1)
                board.push(move1)
                outcome = board.outcome()

            if move2 is not None:
                #     # if game ended, add information for bot2 before stopping loop
                if outcome is not None:
                    reward2 = get_reward(
                        last_board2, board, last_board1, move2, move1, white=bot2.w
                    )
                    pov_board = board if board.turn else board.mirror()
                    replay_buffer.append(
                        (features2, checked_feature2, reward2, pov_board)
                    )
                    pov_board.push_uci(
                        "0000"
                    )  # push null move for bot2, so that it's the turn of bot1 again
                    factor = -1 if (abs(reward2) == reward_spec["win"]) else 1
                    replay_buffer.append(
                        (
                            features1,
                            checked_feature1,
                            factor * reward2,
                            pov_board.mirror(),
                        )
                    )
                    break

            last_board2 = board.copy()
            action2, features2, checked_feature2 = bot2.play(
                board, e=e / 2, training=True
            )
            move2 = board.parse_uci(action2)
            board.push(move2)
            outcome = board.outcome()

            if train_white:
                if move2 is not None:
                    reward1 = get_reward(
                        last_board1, board, last_board2, move1, move2, white=bot1.w
                    )
                    pov_board = board if board.turn else board.mirror()
                    replay_buffer.append((features1, checked_feature1, reward1, board))

                    # if game ended, add other perspective before stopping the loop
                    if outcome is not None:
                        factor = -1 if (abs(reward1) == reward_spec["win"]) else 1
                        pov_board.push_uci(
                            "0000"
                        )  # push null move for bot1, so that it's the turn of bot2 again
                        replay_buffer.append(
                            (
                                features2,
                                checked_feature2,
                                factor * reward1,
                                pov_board.mirror(),
                            )
                        )
                        break
            train_white = True  # if train_white was False, bot2 starts

            if len(replay_buffer) > 500:
                # minibatch training
                targets = []
                features = []
                checked_feature = []
                minibatch = random.choices(replay_buffer, k=58) + replay_buffer[-8:]
                for tup in minibatch:
                    m, x1, x2 = bot1.get_best_action(tup[3], action_vfa, e=0)
                    if m is not None:
                        q_vfa_old = evaluation_vfa(x1, x2)
                        y = torch.Tensor([tup[2]]) + gamma * q_vfa_old
                    else:
                        y = torch.Tensor([[tup[2]]])

                    targets.append(y)
                    features.append(tup[0])
                    checked_feature.append(tup[1])

                targets = torch.cat(targets)
                features = torch.cat(features, dim=0)
                checked_feature = torch.cat(checked_feature, dim=0)
                preds = action_vfa(features, checked_feature)

                adam.zero_grad()
                loss = loss_fn(preds, targets)
                loss.backward()
                adam.step()

                # ****** progress monitoring ******
                iter_count += 1
                c_counter += 1

                if iter_count % 20 == 0:
                    end = time.time()
                    diff = round(end - start, 6)
                    print(f"iter: {iter_count}, loss: {loss}, took {diff} seconds")
                    start = time.time()
                ## after certain number of iteration: update vfa params for evaluation
                if c_counter % C == 0:
                    print("*********** new weights EVALUATION ***********")
                    evaluation_vfa.load_state_dict(action_vfa.state_dict())
                    for param in evaluation_vfa.parameters():
                        param.requires_grad = False
                    for ad in adam.param_groups:
                        ad["lr"] = learning_rate

                avg_loss.append(loss)
            else:
                print("replaybuffer:", len(replay_buffer))

        print(f"Game {g} outcome: {outcome}")
        if len(replay_buffer) > 500:
            last_five_games.append(outcome.winner == int(bot1.w))

            if g % 10 == 0:
                save_data(board.move_stack, f"out/games/{run}/game_{g}.dat")
                torch.save(
                    action_vfa.state_dict(),
                    f"out/models/{run}/m_{g}_iter_{iter_count}.ckpt",
                )

            logger.add_scalar("loss", sum(avg_loss) / len(avg_loss), g)
            logger.add_scalar("number_moves", len(board.move_stack), g)
            logger.add_scalar("number_living", len(board.piece_map()), g)
            logger.flush()

            scheduler.step()


if __name__ == "__main__":
    train(
        run="v8d",
        action_vfa_file=None,
        evaluation_vfa_file=None,
        opponent_vfa_file=None,
    )
