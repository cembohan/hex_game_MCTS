import argparse
import importlib
import sys
import inspect

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

def instantiate_agent(agent_class, colour, temperature):
    """Safely initializes an agent, passing temperature only if supported."""
    sig = inspect.signature(agent_class.__init__)
    if 'temperature' in sig.parameters:
        return agent_class(colour, temperature=temperature)
    return agent_class(colour)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hex",
        description="Run a game of Hex. By default, two naive agents will play.",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 1 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        default="Red",
        type=str,
        help="Specify the player 1 name",
    )
    parser.add_argument(
        "-t1",
        "--temp1",
        type=float,
        default=0.1,
        help="MCTS Temperature for Player 1 (exploration rate)",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 2 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        default="Blue",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-t2",
        "--temp2",
        type=float,
        default=0.1,
        help="MCTS Temperature for Player 2 (exploration rate)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Specify the board size",
    )
    parser.add_argument(
        "-turns",
        "--max_turns",
        type=int,
        default=None,
        help="Maximum number of turns to play before terminating the game",
    )


    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)
    p1_agent_class = getattr(p1, p1_class)
    p2_agent_class = getattr(p2, p2_class)
    g = Game(
        player1=Player(
            name=args.player1Name,
            agent=instantiate_agent(p1_agent_class, Colour.RED, args.temp1),
        ),
        player2=Player(
            name=args.player2Name,
            agent=instantiate_agent(p2_agent_class, Colour.BLUE, args.temp2),
        ),
        board_size=args.board_size,
        max_turns=args.max_turns,
        logDest=args.log,
        verbose=args.verbose,
    )
    g.run()
