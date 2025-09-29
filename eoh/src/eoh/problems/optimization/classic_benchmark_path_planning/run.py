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

from queue import Empty

def _worker_dynamic(import_string, code_string, bench_kwargs, out_q):
    try:
        planning_module = types.ModuleType("planning_module")
        exec(import_string + code_string, planning_module.__dict__)
        sys.modules[planning_module.__name__] = planning_module

        planner = planning_module.Planner(max_iter=5000)
        # bench_kwargs 안에 bench_fn 넣어줬다고 가정
        bench_fn = bench_kwargs["bench_fn"]
        result = bench_fn(planner.plan)
        out_q.put(result)
    except Exception as e:
        out_q.put(e)

def evaluate_with_timeout_dynamic(import_string, code_string, bench_fn, timeout):
    ctx = mp.get_context()  # spawn/fork 둘 다 가능
    q = ctx.Queue()
    bench_kwargs = {"bench_fn": bench_fn}

    p = ctx.Process(
        target=_worker_dynamic,
        args=(import_string, code_string, bench_kwargs, q)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate(); p.join()
        q.close(); q.join_thread()
        print("Timeout or error during evaluation.")
        return None, None

    try:
        payload = q.get_nowait()
    except Empty:
        q.close(); q.join_thread()
        return None, None

    q.close(); q.join_thread()
    if isinstance(payload, Exception):
        raise payload
    return payload

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
            
        self.time_out = 40.0

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


    def __evaluate_path(self, code_string) -> float:        
        result, avg_result = evaluate_with_timeout_dynamic(
            self.import_string,
            code_string,
            self.benchmarker.run,
            timeout=self.time_out
        )

            
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
            if init:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Create a new module object
                    planning_module = types.ModuleType("planning_module")
                    
                    # Execute the code string in the new module's namespace
                    exec(self.import_string+code_string, planning_module.__dict__)

                    # Add the module to sys.modules so it can be imported
                    sys.modules[planning_module.__name__] = planning_module

                    fitness, results = self.__evaluate_initial(planning_module)

            else:
                fitness, results = self.__evaluate_path(code_string=code_string)

            return fitness, results
        
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return None, {"Traceback" : traceback.format_exc()}
