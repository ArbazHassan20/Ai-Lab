import chess
import chess.engine

def evaluate_board(board):
    """Simple evaluation function using Stockfish."""
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        return result['score'].relative.score(mate_score=10000)  

def beam_search(board, beam_width=3, depth_limit=3):
    """Performs beam search to find the best move sequence."""
    beam = [(board, [])]  # (current board state, move sequence)
    
    for _ in range(depth_limit):
        candidates = []
        
        for state, moves in beam:
            for move in state.legal_moves:
                new_board = state.copy()
                new_board.push(move)
                score = evaluate_board(new_board)
                candidates.append((score, new_board, moves + [move]))
        
        candidates.sort(reverse=True, key=lambda x: x[0])  # Sort by score
        beam = [(board, moves) for _, board, moves in candidates[:beam_width]]
    
    best_score, best_board, best_moves = max(candidates, key=lambda x: x[0])
    return best_moves, best_score

board = chess.Board()  # Start with the initial chess position
best_moves, best_score = beam_search(board, beam_width=3, depth_limit=3)
print("Best move sequence:", best_moves)
print("Evaluation score:", best_score)
