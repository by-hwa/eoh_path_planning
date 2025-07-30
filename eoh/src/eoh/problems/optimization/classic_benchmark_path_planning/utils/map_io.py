import pickle
import os
from .architecture_utils import Map

class MapIO:
    @staticmethod
    def save_map(map_data: Map, filename: str) -> None:
        """Save Map object to a binary file."""
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)

    @staticmethod
    def load_map(filename: str) -> Map:
        """Load Map object from a binary file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Map file not found: {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)