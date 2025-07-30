import numpy as np
import importlib
from .utils.prompts import GetPrompts
from .utils.benchmark import MultiMapBenchmarker
from .utils.map_io import MapIO
from .utils.architecture_utils import PlannerResult

import types
import warnings
import sys
import traceback

import copy
import json
import tracemalloc
from typing import TYPE_CHECKING, List, Tuple, Type, Any, Dict, Union, Optional

class PATHPLANNING():
    def __init__(self):
        self.prompts = GetPrompts()
        self.maps = self.__load_maps()
        self.benchmarker = MultiMapBenchmarker(maps=self.maps, iter=10)
        self.ref_result, self.ref_avg_result = self.benchmarker.run(rrt)

    def __load_maps(self):
        maps = []
        maps.append(MapIO.load_map("Multi_obs_map.pkl"))
        maps.append(MapIO.load_map("Maze_map_easy.pkl"))
        maps.append(MapIO.load_map("Narrow_map.pkl"))
        return maps
    
    def __load_ref_alg(self, alg_name:str):
        with open("./utils/classic_method.json", "r") as f:
            result_data = json.load(f)
        for ref_alg in result_data:
            if ref_alg['algorithm'] == alg_name:
                return ref_alg['code']


    def __evaluate_path(self, alg) -> float:
        planner = alg.Planner(max_iter=10000)
        result, avg_result = self.benchmarker.run(planner.plan)

        fitness = MultiMapBenchmarker.get_improvement(self.ref_avg_result, avg_result)['objective_score'].sum()

        return -fitness, avg_result

        
    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                planning_module = types.ModuleType("planning_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, planning_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[planning_module.__name__] = planning_module

                fitness, results = self.__evaluate_path(planning_module)

                return fitness, results
        
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return None, {"Traceback" : traceback.format_exc()}
        


