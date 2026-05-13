from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import time
import os
import torch
import traceback
import importlib
import inspect

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

# ── Web config set by Hex_temp.py when launched with --web ──────────────
# Describes what each player slot is. If not set, falls back to legacy
# "Human vs best_model" behaviour.
#
# Structure:
#   web_config = {
#       'board_size': int,
#       'p1': { 'type': 'human' | 'agent', 'module': str, 'class': str,
#               'path': str|None, 'temp': float, 'name': str },
#       'p2': { ... same ... },
#   }
web_config = None


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


def _instantiate_agent(agent_class, colour, temperature, model_path=None):
    """Safely initialise an agent, passing temperature and model_path only if supported."""
    sig = inspect.signature(agent_class.__init__)
    if 'temperature' in sig.parameters or 'model_path' in sig.parameters:
        return agent_class(colour, temperature=temperature, model_path=model_path)
    return agent_class(colour)


def _build_player(slot_cfg, colour, game_id):
    """
    Given a player-slot config dict, return a (Player, is_human) tuple.
    """
    if slot_cfg['type'] == 'human':
        agent = WebAgent(colour, game_id)
        agent.set_state_callback = set_state_callback
        return Player("Human", agent), True

    # It's an AI agent
    mod = importlib.import_module(slot_cfg['module'])
    cls = getattr(mod, slot_cfg['class'])
    agent = _instantiate_agent(cls, colour, slot_cfg.get('temp', 0.1),
                               model_path=slot_cfg.get('path'))
    return Player(slot_cfg.get('name', slot_cfg['class']), agent), False


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)


@app.route('/config', methods=['GET'])
def get_config():
    """Return the server-side config so the frontend can adapt."""
    if web_config:
        return jsonify({
            'board_size': web_config['board_size'],
            'p1_type': web_config['p1']['type'],
            'p1_name': web_config['p1'].get('name', 'Player 1'),
            'p2_type': web_config['p2']['type'],
            'p2_name': web_config['p2'].get('name', 'Player 2'),
            'p1_temp': web_config['p1'].get('temp', 0.1),
            'p2_temp': web_config['p2'].get('temp', 0.1),
        })
    # Legacy fallback
    return jsonify({
        'board_size': 11,
        'p1_type': 'human',
        'p1_name': 'Human',
        'p2_type': 'agent',
        'p2_name': 'Agent1',
        'p1_temp': 0.1,
        'p2_temp': 0.1,
    })


@app.route('/start', methods=['POST'])
def start_game():
    global active_game_id, active_web_agent, active_game
    active_game_id += 1
    my_game_id = active_game_id

    data = request.json or {}

    # Kill old game if exists
    if active_web_agent is not None:
        try:
            active_web_agent.move_queue.put(None)
        except Exception:
            pass

    active_web_agent = None

    # ── Determine player setup ───────────────────────────────────────
    if web_config:
        board_size = web_config['board_size']

        # Allow the frontend to override temperature via the slider
        p1_cfg = dict(web_config['p1'])
        p2_cfg = dict(web_config['p2'])
        if 'temperature' in data:
            # Legacy: single-slider override applies to all AI players
            p1_cfg['temp'] = float(data['temperature'])
            p2_cfg['temp'] = float(data['temperature'])

        # Allow the frontend to swap human colour in Human-vs-AI mode
        human_plays = data.get('human_plays')
        p1_is_human = p1_cfg['type'] == 'human'
        p2_is_human = p2_cfg['type'] == 'human'

        # If exactly one human, allow the frontend to choose colour
        if (p1_is_human ^ p2_is_human) and human_plays:
            if human_plays == 'RED' and not p1_is_human:
                p1_cfg, p2_cfg = p2_cfg, p1_cfg
            elif human_plays == 'BLUE' and not p2_is_human:
                p1_cfg, p2_cfg = p2_cfg, p1_cfg

        p1, p1_human = _build_player(p1_cfg, Colour.RED, my_game_id)
        p2, p2_human = _build_player(p2_cfg, Colour.BLUE, my_game_id)

        if p1_human:
            active_web_agent = p1.agent
        elif p2_human:
            active_web_agent = p2.agent

    else:
        # ── Legacy behaviour (no --web config) ───────────────────────
        human_plays = data.get('human_plays', 'RED')
        temperature = float(data.get('temperature', 0.1))

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
        state['board_size'] = board.size

        # Tell the frontend which colour the human plays (if any)
        p1_name = getattr(active_game.players.get(Colour.RED), 'name', None)
        p2_name = getattr(active_game.players.get(Colour.BLUE), 'name', None)
        if p1_name == "Human":
            state['human_colour'] = "RED"
        elif p2_name == "Human":
            state['human_colour'] = "BLUE"
        else:
            state['human_colour'] = None  # AI vs AI

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