import pygame 
import chess
import time
from bot.bots import HaiBotLong

pygame.init()
pygame.font.init() 
myfont = pygame.font.SysFont('Comic Sans MS', 40)

screen = pygame.display.set_mode((800, 800))

chessboard = pygame.image.load('icons/chessboard.png')
selection = pygame.image.load('icons/selection-box.png')
chess_icons = dict()
chess_icons["K"] = pygame.image.load('icons/white_king.png')
chess_icons["Q"] = pygame.image.load('icons/white_queen.png')
chess_icons["B"] = pygame.image.load('icons/white_bishop.png')
chess_icons["N"] = pygame.image.load('icons/white_knight.png')
chess_icons["R"] = pygame.image.load('icons/white_rook.png')
chess_icons["P"] = pygame.image.load('icons/white_pawn.png')
chess_icons["k"] = pygame.image.load('icons/black_king.png')
chess_icons["q"] = pygame.image.load('icons/black_queen.png')
chess_icons["b"] = pygame.image.load('icons/black_bishop.png')
chess_icons["n"] = pygame.image.load('icons/black_knight.png')
chess_icons["r"] = pygame.image.load('icons/black_rook.png')
chess_icons["p"] = pygame.image.load('icons/black_pawn.png')

def map_chess_square_coordinates(chess_square): 
    x = ((chess_square%8)) * 100
    y = (7 - (chess_square//8)) * 100 
    return (x, y)

def map_coordinates_chess_square(x, y): 
    chess_square = (x//100) + (7 - y//100) * 8
    return chess_square


def draw_board(board, screen): 
    for chess_square, piece in  board.piece_map().items(): 
        screen.blit(chess_icons[str(piece)], map_chess_square_coordinates(chess_square))


def check_selected_square(selected_square, board): 
    if selected_square is None: 
        result = None
    else: 
        result = board.piece_at(selected_square)
    
    if result is None: 
        return False
    else: 
        return str(result).isupper()

def promotion(move, board): 
    pawn = str(board.piece_at(move[0])).lower() == "p"
    end = move[1] in range(56, 64)
    if pawn & end: 
        return 5


def update_board(move, board, last_move): 
    """Makes move and updates the screen."""
    promote = promotion(move, board)
    move_uci = chess.Move(*move, promotion=promote).uci()
    try: 
        board.push_uci(move_uci)
        last_move[0] = move[0]
        last_move[1] = move[1]
        move[0] = None
        move[1] = None
        return True
    except ValueError: 
        move[1] = None
        return False


def bot_play_wrapper(bot, e=0.02): 
    def bot_play(board, last_move):
        outcome = board.outcome() 
        if outcome is None:  
            action_bot = bot.play(board, e=e, training=False)
            move_bot = board.parse_uci(action_bot)
            board.push(move_bot)
            last_move[0] = move_bot.from_square
            last_move[1] = move_bot.to_square
            outcome = board.outcome()
        return outcome
    return bot_play


def check_outcome(outcome, screen): 
    if outcome is not None: 
        if outcome.winner is None: 
            text = "You are lucky to have played draw with HaiBotLong."
        elif outcome.winner: 
            text = "HaiBotLong has let you won."
        else: 
            text = "HaiBotLong has destroyed you."

        textsurface = myfont.render(text, False, (255, 128, 0), (250, 250,250))
        text_rect = textsurface.get_rect(center=(400, 400))
        screen.blit(textsurface,text_rect)
        textsurface2 = myfont.render("press R to restart", False, (255, 128, 0), (250, 250,250))
        screen.blit(textsurface2,(0,0))


def mark_in_check(in_check_list, board, screen): 
    in_check_list.clear()
    if board.is_check(): 
        s = pygame.Surface((100,100), pygame.SRCALPHA) 
        s.fill((255,0,0,50))      
        square_set = board.pieces(6, board.turn)
        for square in square_set: 
            screen.blit(s, map_chess_square_coordinates(square)) 


def mark_last_move(last_move, screen): 
    s = pygame.Surface((100,100), pygame.SRCALPHA) 
    s.fill((0,255,0,50))      
    for square in last_move:   
        if square is not None: 
            screen.blit(s, map_chess_square_coordinates(square)) 


def main():             
    bot_path = "out/models/v8/m_280_iter_23450.ckpt"
    bot = HaiBotLong(color="B", model_path=bot_path)
    bot_play = bot_play_wrapper(bot)

    move = [None, None]
    opponent_turn = False
    outcome = None
    in_check_list = []
    last_move = [None, None]
    board = chess.Board()


    running = True
    while running: 

        if opponent_turn: 
            time.sleep(0.5)
            outcome = bot_play(board, last_move)
        opponent_turn = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    running = main()
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                selected_square = map_coordinates_chess_square(*pos)

                if selected_square == move[0]: 
                    selected_square = None
                    move[0] = None
                elif check_selected_square(selected_square, board): 
                    move[0] = selected_square
                elif move[0] is not None: 
                    move[1] = selected_square
                    if update_board(move, board, last_move): 
                        opponent_turn = True


        screen.fill((0, 0, 0))
        screen.blit(chessboard, (0, 0))
        draw_board(board, screen)

        if check_selected_square(move[0], board): 
            screen.blit(selection, map_chess_square_coordinates(move[0]))

        check_outcome(outcome, screen)
        mark_in_check(in_check_list, board, screen) 
        mark_last_move(last_move, screen)

        pygame.display.update()

    return False


if __name__ == "__main__": 
    main()
    