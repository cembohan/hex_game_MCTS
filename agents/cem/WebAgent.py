from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move

class WebAgent(AgentBase):
    get_move_callback = None
    set_state_callback = None

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn, board, opp_move):
        if WebAgent.set_state_callback:
            WebAgent.set_state_callback(turn, board, self.colour)
        if WebAgent.get_move_callback:
            return WebAgent.get_move_callback()
        return Move(0, 0) # Fallback
