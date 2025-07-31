import numpy as np
from typing import Tuple, Literal, Union, Optional, List, Dict, NamedTuple, Callable, Any

class Node:
    def __init__(self, position, parent=None, cost=0.0):
        self.position = position        # Tuple[float, ...] → 2D: (x,y), 3D: (x,y,z)
        self.parent = parent            # Node or None
        self.cost = cost                # Path cost
        self.children = []
        self.valid = True               # For collision checking etc.

class Map(NamedTuple):
    grid: np.ndarray
    start: Union[Tuple[float, float], Tuple[float, float, float]]
    goal: Union[Tuple[float, float], Tuple[float, float, float]]
    obstacles: List[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]] # x, y, width, height or x, y, z, width, height, dimension
    size: Union[Tuple[int, int], Tuple[int, int, int]]

class PlannerResult(NamedTuple):
    success: bool                       # Path navigation success or not
    path: List[Tuple[float, ...]]       # Final path from start to goal
    nodes: List[Node]                   # All explored nodes
    edges: List[Tuple[Node, Node]]      # Parent-child connections