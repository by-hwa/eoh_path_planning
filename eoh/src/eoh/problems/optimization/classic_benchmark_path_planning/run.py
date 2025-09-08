import numpy as np
import importlib
from .utils.prompts import GetPrompts
from .utils.benchmark import MultiMapBenchmarker
from .utils.map_io import MapIO
from .utils.architecture_utils import Map
from .utils.architecture_utils import PlannerResult

import types
import warnings
import sys
import os
import traceback

import copy
import json
import tracemalloc
import pandas as pd

from typing import Tuple, Literal, Union, Optional, List, Dict, NamedTuple, Callable, Any, Set, TYPE_CHECKING, Type
# sys.path.append("C:/Workspace/EoH_Path_planning/eoh/src/eoh/problems/optimization/classic_benchmark_path_planning/utils")

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import multiprocessing as mp

import concurrent.futures

class PATHPLANNING():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts = GetPrompts()
        self.maps = self.__load_maps()
        self.import_string = '''
from typing import Tuple, Literal, Union, Optional, List, Dict, NamedTuple, Callable, Any, Set, TYPE_CHECKING, Type
import time
from queue import Queue
import numpy as np
import random
import math
import sys
import os

from eoh.problems.optimization.classic_benchmark_path_planning.utils.architecture_utils import PlannerResult, Map
'''
        self.benchmarker = MultiMapBenchmarker(maps=self.maps, iter=10)
        _, self.ref_avg_result = self.evaluate(self.__load_ref_alg("RRT"), init=True)
        print(self.ref_avg_result)

        filename = "./results/pops/reference_result.json"
        with open(filename, "w") as f:
            json.dump(self.ref_avg_result.to_dict(orient='records'), f)
            f.write("\n")
            
        self.time_out = 150.0

    def __load_maps(self):
        maps_dir = os.path.join(self.base_dir, 'utils', 'maps')

        maps = []
        maps.append(MapIO.load_map(os.path.join(maps_dir, "Multi_obs_map.pkl")))
        maps.append(MapIO.load_map(os.path.join(maps_dir, "Maze_map_easy.pkl")))
        maps.append(MapIO.load_map(os.path.join(maps_dir, "Narrow_map.pkl")))
        return maps
    
    def __load_ref_alg(self, alg_name:str):
        alg_dir = os.path.join(self.base_dir, 'utils', 'classic_method.json')
        with open(alg_dir, "r") as f:
            result_data = json.load(f)
        for ref_alg in result_data:
            if ref_alg['algorithm'] == alg_name:
                return ref_alg['code']


    def __evaluate_path(self, alg) -> float:
        planner = alg.Planner(max_iter=5000)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                future = executor.submit(self.benchmarker.run, planner.plan)
                result, avg_result = future.result(timeout=self.time_out)
            except:
                executor.shutdown(wait=False, cancel_futures=True)
                return None, None

        # with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            # result, avg_result = self.benchmarker.run(planner.plan)
            # future = executor.submit(self.benchmarker.run, planner.plan)
            # result, avg_result = future.result(timeout=self.time_out)
            
        if result is None and avg_result is None:
            return None, None

        improvement = MultiMapBenchmarker.get_improvement(self.ref_avg_result, avg_result)
        fitness = improvement['objective_score'].mean()
        avg_result = pd.concat([avg_result, improvement], axis=1)
        return -fitness, avg_result
    
    def __evaluate_initial(self, alg) -> float:
        planner = alg.Planner(max_iter=5000)
        result, avg_result = self.benchmarker.run(planner.plan)

        return None, avg_result

    def evaluate(self, code_string, init = False):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                planning_module = types.ModuleType("planning_module")
                
                # Execute the code string in the new module's namespace
                exec(self.import_string+code_string, planning_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[planning_module.__name__] = planning_module

                if not init:
                    fitness, results = self.__evaluate_path(planning_module)
                else:
                    fitness, results = self.__evaluate_initial(planning_module)

                return fitness, results
        
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return None, {"Traceback" : traceback.format_exc()}
