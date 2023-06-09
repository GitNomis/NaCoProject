from typing import List
from enum import Enum

class State(Enum):
    """
    State Enum of a cells state with in cellular automata
    Returns:
        State: grid cell state
    """
    BARREN = 0
    FIRE = 1
    WATER = 2
    TREE = 3
    
    @classmethod
    def list(cls) -> List:
        """Return all values of the class ```cls```.

        Returns:
            List: All values within  the class ```cls```.
        """        
        return list(map(lambda c: c.value, cls))
    
    def __int__(self):
        return self.value