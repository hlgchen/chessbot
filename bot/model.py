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
    Takes features describing the current board and board after making a particular move
    and returns an estimated value for that board and move. 
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
        """
        params: 
            - x(torch.Tensor): shape (batch_size, 38,8,8)
            - checked(torch.Tensor): shape (batch_size, 4)
        returns: 
            - x(torch.Tensor): shape (batch_size, 1) containing the value estimation for board move combo
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
    """Takes chess.board before move and chess.board after a move 
    returns features for the value function approximator."""
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
        """Takes FEN string and returns shape (64, 1) numpy integer array 
        with mapping of pieces via self.chess_map."""
        fen = ''.join([self.chess_map.get(piece, piece) for piece in fen])
        fen = np.array([y.split(" ") for y in fen.split('\n')]).astype(np.int)
        return fen.reshape(-1, 1)


    def get_base_features(self, int_array): 
        """Takes shape (64, 1) piece position numpy array and 
        returns torch.Tensor shwoing position of different piece types.
        
        params: 
            - int_array(np.array): shape (64, 1) array containing position of each piece, 
                                    mapping via self.chess_map

        returns: 
            - x_cat(torch.Tensor): shape (1, 16, 8, 8 ) Tensor. 
                                Layer: 
                                    0: k position (indicated with a 1 in (8,8) map)
                                    1:q, 2:b, 3:n, 4:r, 5:p
                                    6: empty fields
                                    7:P, 8:R, 9:N, 10:B, 11:Q, 12:K
                                    13: integer encoding of pieces but in spacial 8,8 map
                                    14: 1 for each black piece
                                    15: 1 for each white piece
        """

        x1 = torch.Tensor(self.one_hot.transform(int_array).todense())
        x1 = x1.reshape(-1,8,8,13).permute(0, 3, 1, 2)

        x2 = torch.Tensor(int_array).reshape(-1,8,8,1).permute(0, 3, 1, 2)
        x3 = (x2 < 0).float()
        x4 = (x2 > 0).float()

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return x_cat
        

    def get_conv_map(self, chess_squares): 
        """Takes list of chess squares and returns torch tensor with shape (1, 1, 8, 8) 
        with number of chess square occurence in spacial map. (E.g. if [1, 1, 2]), there will be a 2 for
        chess square 1 and a 1 for chess square 2, 0 for all other chess squares. 

        Chessquare mapping as follows: 0-A1, 1-B1, 2-C1, 56-A8.

        params: 
            - chess_squares(list): list of integer of chess squares
        returns: 
            - torch.Tensor of shape (1,1,8,8)
        """
        J = np.array(chess_squares)
        I = np.zeros(J.shape)
        V = np.ones(J.shape)
        mat = np.array(sparse.coo_matrix((V,(I,J)),shape=(1,64)).todense())
        mat = np.flip(mat.reshape(8, 8), 0).copy()
        return torch.Tensor(mat).unsqueeze(0).unsqueeze(0)


    def get_extra_features(self, board): 
        """Takes a board and returns more chess specific features.
        Layer0 shows the position of opposing chess pieces that could attack you pieces.
        Layer1 shows the position of your pieces that are under attack by the opponent.
        Layer2 shows the position of opp. pieces that you can attack.
        
        params: 
            - board(chess.Board): current board for which to get the features
        returns: 
            - torch.Tensor of shape (1, 3, 8, 8)
        """
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
        """Takes a board, simulates possible moves of the opponent and returns
        whether the opponent can make a check, checkmate and which pieces these are. 
        
        params: 
            - board(chess.Board)
        returns: 
            - opp_check(bool): True if oppent can give at least one check 
            - opp_checkmate(bool): True if there is one move that would result in a checkmate
            - opp_movements(torch.Tensor): shape(1, 3, 8, 8). 
                layer0: current position of pieces that after a move would give a check
                layer1: positions of pieses when they give a check (after the move)
                layer2: opp_to all possible legal ending postions of the opponent.        
            
        """
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
        opp_movements = torch.cat([opp_checkers_from, opp_checkers_to, opp_to], dim=1)

        return opp_check, opp_checkmate, opp_movements



    def encode_board_move(self, board_list): 
        """Takes list of board tuples and returns feature tensors. 
        params: 
            - board_list(list): list of board tuples. 
                                Tuple should contain current chess.Board at position 0
                                and chess.Board after a certain move at position 1. 
                                If board_list is a single tuple that's ok too (e.g. (board1, board2))
        returns: 
            - x(torch.Tensor): main features with dimension (len(board_list), 38, 8, 8)
            - checked(torch.Tensor): features indicating information about checks and checkmates, 
                                    shape (len(board_list), 4)
        
        """

        if not isinstance(board_list, list): 
            board_list = [board_list]

        x_ls = []
        checked = []
        for board_tuple in board_list:
            int_array1 = self.get_state_features(str(board_tuple[0]))
            int_array2 = self.get_state_features(str(board_tuple[1]))

            x_cat1 = self.get_base_features(int_array1)
            x_cat2 = self.get_base_features(int_array2)
            x_cat3 = self.get_extra_features(board_tuple[1])
            oc, ocm, ocheckers = self.get_one_step_ahead(board_tuple[1])

            x_cat = torch.cat([x_cat1, (x_cat2-x_cat1), x_cat3, ocheckers], dim=1)
            x_ls.append(x_cat)


            checked.append(int(board_tuple[1].is_check()))
            checked.append(int(board_tuple[1].is_checkmate()))
            checked.append(int(oc))
            checked.append(int(ocm))


        checked = torch.Tensor(checked).reshape(-1, 4)
        x = torch.cat(x_ls, dim=0)

        return x, checked