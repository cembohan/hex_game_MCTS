from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move
import queue

class WebAgent(AgentBase):
    def __init__(self, colour: Colour, game_id: int):
        super().__init__(colour)
        self.game_id = game_id
        self.move_queue = queue.Queue()
        self.set_state_callback = None

            # -----------------------------------------------------------------
    def __getstate__(self):
        """Called by pickle/deepcopy – remove what can't be pickled."""
        state = self.__dict__.copy()
        state['move_queue'] = None          # queue contains a lock
        state['set_state_callback'] = None  # function reference can be problematic
        return state

    def __setstate__(self, state):
        """Called after pickle/deepcopy – restore the queue."""
        self.__dict__.update(state)
        self.move_queue = queue.Queue()
        # set_state_callback will be set again by the game later
    # -----------------------------------------------------------------

    def make_move(self, turn, board, opp_move):
        if self.set_state_callback:
            self.set_state_callback(turn, board, self.colour, self.game_id)
        
        move = self.move_queue.get()
        if move is None:
            raise Exception("Game aborted by starting a new game.")
        
        return move
