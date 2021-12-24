import numpy as np 
import torch 
from .model import VFA, BoardMoveTransformer


class HaiBotLong():
    def __init__(self, color, model_path = None): 
        self.w = color == "W"
        self.vfa = VFA()
        self.board_move_transformer = BoardMoveTransformer()
        self.move_mirror_dict = {
            "1": "8", 
            "2": "7", 
            "3": "6", 
            "4": "5", 
            "5": "4", 
            "6": "3", 
            "7": "2", 
            "8": "1"
        }
        if model_path is not None: 
            self.load_vfa_params(torch.load(model_path))


    def load_vfa_params(self, model_state): 
        """Updates model parameters of self.vfa with model_state."""
        self.vfa.load_state_dict(model_state)
        for param in self.vfa.parameters(): 
            param.requires_grad = False


    def select_move(self, values, action_space, e): 
        """
        Samples a move from the action_spce, given values calculated by vfa and e. 

        Params: 
            - values(torch.tensor): tensor with shape (n_actions, 1), contains 
                                    value estimation of state if action was taken
            - action_space(list): actions that can be taken 
                                    (that is uci moves as strings in the list)
            - e (float): value between 0 and 1 for e-greedy. With probability e
                            a random move is chosen.
        Return: 
            move(string): uci move. With e probability random move, otherwise move 
                            with highest value in values. 

        """
        top3 = values.topk(min(3, values.shape[0]), dim=0)[0].min()
        max_mask = values.where(values > top3, torch.Tensor([0])).double().detach()
        max_mask = (max_mask - max_mask.min())**4
        # max_mask = (values == values.max()).double()

        p = torch.nan_to_num(max_mask/max_mask.sum(), 0) * (1-e)
        p = p + e * (1/len(action_space))
        p /= p.sum()
        move_id = np.random.choice(range(len(action_space)), 1, p=p)[0]

        return move_id

    def get_best_action(self, board, vfa, e=0): 
        """
        Returns a move given the board. 

        Params: 
            - _board(chess.board): current board, should be the turn of this bot
            - vfa(VFA): value function approximator to be used for selecting the best action
            - e(float): value between 0 and 1, controls e-greedy 

        Returns: 
            - move: uci move. With e probability random move, otherwise move 
                    with highest value in values. 
        
        
        """
        action_space = [str(m) for m in board.legal_moves] 
        board_list = []
        for move in action_space:
            hypo_board = board.copy() 
            hypo_board.push_uci(move)
            board_list.append(
                (board, hypo_board)
                )

        move = None
        features =None
        checked_feature=None
        if len(action_space) > 0: 
            x, checked = self.board_move_transformer.encode_board_move(board_list)
            values = vfa(x, checked).squeeze(dim=-1)
            if e > 0: 
                move_id = self.select_move(values, action_space, e)
            else: 
                move_id = values.argmax().item()

            move = action_space[move_id]
            features = x[move_id].unsqueeze(0)
            checked_feature = checked[move_id].unsqueeze(0)
        return move, features, checked_feature

    
    def get_value_best_move(self, board, vfa): 
        """
        For Double Q-learning, use older Q network for evaluation (self.vfa)
        Takes board and vfa for action seclection. Selects the action with highest 
        value accoring to vfa and returns value estimation of self.vfa .
        """
        move, x, checked_feature = self.get_best_action(board, vfa)
        if move is None: 
            return None
        else: 
            return self.vfa(x, checked_feature)

    
    def mirror_move(self, move): 
        """Take uci move as string and mirror the move."""
        move = [self.move_mirror_dict.get(l, l) for l in move]
        return "".join(move)


    def play(self, _board, e=0.1, vfa=None, training=False):
        """"Make a play based on the value function approximationa and current board. 
        Returns the move to be made. 

        Params:
            - _board(chess.board): current board, should be the turn of this bot
            - e(float): value between 0 and 1, controls e-greedy 
            - vfa(VFA): value function approximator to be used for selecting the best action

        Returns: 
            - move (string): uci move as a string
        
        """
        if vfa is None: 
            vfa = self.vfa
        board = _board.copy() if self.w else _board.mirror()
        move, features, checked_feature = self.get_best_action(board, vfa, e)
        move = move if self.w else self.mirror_move(move)

        if not training: 
            return move
        else: 
            return move, features, checked_feature


