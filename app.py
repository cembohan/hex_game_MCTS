from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import queue
import time
import os

from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from agents.cem.WebAgent import WebAgent

# Import Agent1 and NaiveAgent just in case it is needed.
# The user wants Agent1 as the AI.
try:
    from agents.cem.agent1 import Agent1
except ImportError:
    pass

app = Flask(__name__, static_folder='static')
CORS(app)

game_state = {
    'board': None,
    'turn': 0,
    'waiting_for_human': False,
    'human_colour': None,
    'game_over': False,
    'winner': None
}

move_queue = queue.Queue()

def set_state_callback(turn, board, colour):
    # board is a Board object. Serialize it for the web.
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
    
    game_state['board'] = board_data
    game_state['turn'] = turn
    game_state['waiting_for_human'] = True
    game_state['human_colour'] = 'RED' if colour == Colour.RED else 'BLUE'

def get_move_callback():
    move = move_queue.get()
    game_state['waiting_for_human'] = False
    return move

WebAgent.set_state_callback = set_state_callback
WebAgent.get_move_callback = get_move_callback

current_game_thread = None

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.route('/start', methods=['POST'])

def start_game():
    data = request.json
    human_plays = data.get('human_plays', 'RED')
    temperature = float(data.get('temperature', 0.1))

    # Make sure we use Agent1 as requested
    if human_plays == 'RED':
        p1 = Player("Human", WebAgent(Colour.RED))
        p2 = Player("Agent1", Agent1(Colour.BLUE, temperature=temperature))
    else:
        p1 = Player("Agent1", Agent1(Colour.RED, temperature=temperature))
        p2 = Player("Human", WebAgent(Colour.BLUE))

    game_state['board'] = None
    game_state['turn'] = 0
    game_state['waiting_for_human'] = False
    game_state['game_over'] = False
    game_state['winner'] = None

    # Clear move queue
    while not move_queue.empty():
        move_queue.get()

    def run_game():
        # Using a fixed 11x11 board
        g = Game(p1, p2, board_size=11, logDest="web_game.log", verbose=True)
        try:
            result = g.run()
            game_state['winner'] = result.get('winner', 'Draw')
        except Exception as e:
            game_state['winner'] = f"Error: {e}"
        finally:
            game_state['game_over'] = True
            
            # Update final board state
            board = g.board
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
            game_state['board'] = board_data

    global current_game_thread
    current_game_thread = threading.Thread(target=run_game, daemon=True)
    current_game_thread.start()

    return jsonify({"status": "started"})

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(game_state)

@app.route('/move', methods=['POST'])
def make_move():
    if not game_state['waiting_for_human']:
        return jsonify({"error": "Not human turn"}), 400
    
    data = request.json
    if data.get('swap'):
        from src.Move import Move
        move_queue.put(Move(-1, -1))
        return jsonify({"status": "ok"})
    
    x = data.get('x')
    y = data.get('y')
    if x is not None and y is not None:
        from src.Move import Move
        move_queue.put(Move(int(x), int(y)))
        return jsonify({"status": "ok"})
    
    return jsonify({"error": "Invalid move"}), 400

if __name__ == '__main__':
    # Ensure static dir exists
    os.makedirs('static', exist_ok=True)
    print("Starting Web Server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
