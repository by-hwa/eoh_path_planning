import numpy as np
import importlib
from .prompts import GetPrompts
import types
import warnings
import sys

class PATHPLANNING():
    def __init__(self):
        self.prompts = GetPrompts()


    def online_binpack(self,items: tuple, bins: np.ndarray, alg):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        n = 1
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
            n=n+1
            
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins


    def evaluateGreedy(self,alg) -> float:


        return fitness

        
    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateGreedy(heuristic_module)

                return fitness
        except Exception as e:
            #print("Error:", str(e))
            return None




