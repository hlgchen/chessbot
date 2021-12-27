import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import chess

class VFA(nn.Module): 
    """
    Value function approximation. 
    Takes a (list of) board-states (fen-string) and outputs the value of the state.
    https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation 
    """
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.ModuleList(
            [
                nn.Conv2d(38, 3, kernel_size=i, padding='same')
                for i in range(1, 9)
            ]
        )
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 16, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(16)
        self.l1 = nn.Linear(404, 32)
        self.l2 = nn.Linear(32, 1)


    def forward(self, x, checked): 
        """Takes board and move tensor representation, returns the value. 
        """

        x = torch.cat([self.conv1[i](x) for i in range(8)], dim=1)
        x = self.bn1(x)
        x = self.conv2(F.leaky_relu(x))
        x = self.bn2(x)
        x = F.leaky_relu(x).flatten(start_dim=1)
        x = torch.cat([x, checked], dim=1)
        x = self.l1(x)
        x = self.l2(F.leaky_relu(x))
        return x


class BoardMoveTransformer(): 
    def __init__(self): 
        super().__init__()
        self.chess_map = {
            "k": "-6", 
            "q": "-5", 
            "b": "-4", 
            "n": "-3", 
            "r": "-2", 
            "p": "-1", 
            ".": "0", 
            "P": "1", 
            "R": "2", 
            "N": "3", 
            "B": "4", 
            "Q": "5", 
            "K": "6", 
        }

        self.one_hot = OneHotEncoder()
        self.one_hot.fit(self.get_state_features(str(chess.Board())))


    def get_state_features(self, fen): 
        """Takes FEN string and returns shape (64, 1) numpy string array 
        with mapping of pieces via self.chess_map."""
        fen = ''.join([self.chess_map.get(piece, piece) for piece in fen])
        fen = np.array([y.split(" ") for y in fen.split('\n')]).astype(np.int)
        return fen.reshape(-1, 1)


    def get_base_features(self, int_array): 

        x1 = torch.Tensor(self.one_hot.transform(int_array).todense())
        x1 = x1.reshape(-1,8,8,13).permute(0, 3, 1, 2)

        x2 = torch.Tensor(int_array).reshape(-1,8,8,1).permute(0, 3, 1, 2)
        x3 = (x2 < 0).float()
        x4 = (x2 > 0).float()

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return x_cat
        

    def get_conv_map(self, chess_squares): 
        J = np.array(chess_squares)
        I = np.zeros(J.shape)
        V = np.ones(J.shape)
        mat = np.array(sparse.coo_matrix((V,(I,J)),shape=(1,64)).todense())
        mat = np.flip(mat.reshape(8, 8), 0).copy()
        return torch.Tensor(mat).unsqueeze(0).unsqueeze(0)


    def get_extra_features(self, board): 
        my_pieces = []
        for i in range(1, 7): 
            my_pieces += list(board.pieces(i,chess.WHITE))

        opp_attackers = []
        attacked_pos = []
        attackable_pos = []
        for i in my_pieces: 
            attackers = list(board.attackers(chess.BLACK, i))
            opp_attackers += attackers
            if len(attackers) > 0: 
                attacked_pos.append(i)
            attackable_pos += list(board.attacks(i))
                
        opp_attackers = self.get_conv_map(opp_attackers)
        attacked_pos = self.get_conv_map(attacked_pos)
        attackable_pos = self.get_conv_map(attackable_pos)

        return torch.cat([opp_attackers, attacked_pos, attackable_pos], dim=1)


    def get_one_step_ahead(self, board): 
        opp_check = False
        opp_checkmate = False
        opp_checkers_from = []
        opp_checkers_to = []
        opp_to = []

        for m in board.legal_moves: 
            hypo_board = board.copy()
            hypo_board.push(m)
            if hypo_board.is_check(): 
                opp_check = True
                opp_checkers_to.append(m.to_square)
                opp_checkers_from.append(m.from_square)
                if hypo_board.is_checkmate():
                    opp_checkmate = True
            opp_to.append(m.to_square)

        opp_checkers_from = self.get_conv_map(opp_checkers_from)
        opp_checkers_to = self.get_conv_map(opp_checkers_to)
        opp_to = self.get_conv_map(opp_to)
        opp_checkers = torch.cat([opp_checkers_from, opp_checkers_to, opp_to], dim=1)

        return opp_check, opp_checkmate, opp_checkers



    def encode_board_move(self, board_list): 

        if not isinstance(board_list, list): 
            board_list = [board_list]

        x_ls = []
        checked = []
        for board in board_list:
            int_array1 = self.get_state_features(str(board[0]))
            int_array2 = self.get_state_features(str(board[1]))

            x_cat1 = self.get_base_features(int_array1)
            x_cat2 = self.get_base_features(int_array2)
            x_cat3 = self.get_extra_features(board[1])
            oc, ocm, ocheckers = self.get_one_step_ahead(board[1])

            x_cat = torch.cat([x_cat1, (x_cat2-x_cat1), x_cat3, ocheckers], dim=1)
            x_ls.append(x_cat)


            checked.append(int(board[1].is_check()))
            checked.append(int(board[1].is_checkmate()))
            checked.append(int(oc))
            checked.append(int(ocm))


        checked = torch.Tensor(checked).reshape(-1, 4)
        x = torch.cat(x_ls, dim=0)

        return x, checked