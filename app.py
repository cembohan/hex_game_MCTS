from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import time
import os
import torch
import traceback

from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from agents.cem.WebAgent import WebAgent

try:
    from agents.cem.agent1 import Agent1
except ImportError:
    pass

app = Flask(__name__, static_folder='static')
CORS(app)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

game_state = {
    'turn': 0,
    'waiting_for_human': False,
    'game_over': False,
    'winner': None
}

active_game_id = 0
active_web_agent = None
active_game = None


def set_state_callback(turn, board, colour, game_id):
    if game_id != active_game_id:
        return
    game_state['turn'] = turn
    game_state['waiting_for_human'] = True


def load_model_safe(path, max_retries=5, delay=0.3):
    """
    Try to load a PyTorch checkpoint from `path`.
    Retries several times to survive write collisions during training.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            _ = checkpoint['model_state_dict']   # sanity check
            return checkpoint
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(delay)
    raise RuntimeError(f"Failed to load model from {path} after {max_retries} attempts") from last_exc


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)


@app.route('/start', methods=['POST'])
def start_game():
    global active_game_id, active_web_agent
    active_game_id += 1
    my_game_id = active_game_id

    data = request.json
    human_plays = data.get('human_plays', 'RED')
    temperature = float(data.get('temperature', 0.1))

    # Kill old game if exists
    if active_web_agent is not None:
        try:
            active_web_agent.move_queue.put(None)
        except Exception:
            pass

    # ---- Load the latest model safely ----------
    checkpoint_dir = os.path.join("agents", "cem", "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.isfile(best_model_path):
        return jsonify({"error": "No model checkpoint found. Train the agent first."}), 404

    try:
        checkpoint = load_model_safe(best_model_path)
        model_state = checkpoint['model_state_dict']
        board_size = checkpoint.get('board_size', 11)
    except Exception as e:
        return jsonify({"error": f"Cannot load AI model: {str(e)}"}), 500

    # ---- Create agents -------------------------
    human_agent = WebAgent(Colour.RED if human_plays == 'RED' else Colour.BLUE, my_game_id)
    human_agent.set_state_callback = set_state_callback
    active_web_agent = human_agent

    if human_plays == 'RED':
        p1 = Player("Human", human_agent)
        p2 = Player("Agent1", Agent1(Colour.BLUE,
                                     temperature=temperature,
                                     state_dict=model_state,
                                     board_size=board_size))
    else:
        p1 = Player("Agent1", Agent1(Colour.RED,
                                     temperature=temperature,
                                     state_dict=model_state,
                                     board_size=board_size))
        p2 = Player("Human", human_agent)

    # Reset game state
    game_state['turn'] = 0
    game_state['waiting_for_human'] = False
    game_state['game_over'] = False
    game_state['winner'] = None

    # Run game in background
    def run_game():
        global active_game
        try:
            g = Game(p1, p2, board_size=board_size, logDest="web_game.log", verbose=True)
            active_game = g
            result = g.run()
            if my_game_id == active_game_id:
                game_state['winner'] = result.get('winner', 'Draw')
        except Exception as e:
            print("\n\n==== GAME CRASHED ====")
            traceback.print_exc()
            print("======================\n")
            if my_game_id == active_game_id:
                game_state['winner'] = f"Error: {e}"
        finally:
            if my_game_id == active_game_id:
                game_state['game_over'] = True

    threading.Thread(target=run_game, daemon=True).start()

    # ===== THIS IS THE MISSING LINE =====
    return jsonify({"status": "started"})


@app.route('/state', methods=['GET'])
def get_state():
    state = dict(game_state)
    if active_game is not None:
        board = active_game.board
        board_data = []
        for i in range(board.size):
            row = []
            for j in range(board.size):
                c = board.tiles[i][j].colour
                if c == Colour.RED:
                    row.append('R')
                elif c == Colour.BLUE:
                    row.append('B')
                else:
                    row.append('E')
            board_data.append(row)
        state['board'] = board_data
        state['turn'] = active_game.turn

        if getattr(active_game.players.get(Colour.RED), 'name', None) == "Human":
            state['human_colour'] = "RED"
        else:
            state['human_colour'] = "BLUE"
    return jsonify(state)


@app.route('/move', methods=['POST'])
def make_move():
    if not game_state['waiting_for_human'] or active_web_agent is None:
        return jsonify({"error": "Not human turn"}), 400

    data = request.json
    if data.get('swap'):
        from src.Move import Move
        active_web_agent.move_queue.put(Move(-1, -1))
        game_state['waiting_for_human'] = False
        return jsonify({"status": "ok"})

    x = data.get('x')
    y = data.get('y')
    if x is not None and y is not None:
        from src.Move import Move
        active_web_agent.move_queue.put(Move(int(x), int(y)))
        game_state['waiting_for_human'] = False
        return jsonify({"status": "ok"})

    return jsonify({"error": "Invalid move"}), 400


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("Starting Web Server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)