import numpy as np
import importlib
# from .prompts import GetPrompts
from .prompts_hier import GetPrompts
import types
import warnings
import sys
import traceback
# sys.path.append('C:\Workspace\EoH_Path_planning')
sys.path.append('C:\Workspace\PathBench\src')

from algorithms.configuration.configuration import Configuration
from simulator.services.services import Services
import copy
from simulator.simulator import Simulator
from structures import Point

from algorithms.classic.graph_based.a_star import AStar
from algorithms.classic.testing.a_star_testing import AStarTesting
from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting

import tracemalloc
from typing import TYPE_CHECKING, List, Tuple, Type, Any, Dict, Union, Optional
from algorithms.configuration.maps.map import Map

class PATHPLANNING():
    def __init__(self):
        self.prompts = GetPrompts()
        self.maps = self.load_maps()
        self.a_star_results = self.__get_a_star_results(self.maps)
        self.a_star_statistics = self.__get_results(self.a_star_results)
        self.import_string = '''
from typing import List

import torch
import numpy as np
import random
import time
import math

from memory_profiler import profile
from typing import TYPE_CHECKING, List, Tuple, Type, Any, Dict, Union, Optional

from algorithms.basic_testing import BasicTesting

from algorithms.algorithm import Algorithm
from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest
from algorithms.classic.sample_based.core.graph import gen_cyclic_graph, CyclicGraph

from algorithms.configuration.entities.entity import Entity
from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.entities.obstacle import Obstacle
from algorithms.configuration.entities.trace import Trace
from algorithms.configuration.maps.map import Map
from algorithms.configuration.maps.bresenhams_algo import bresenhamline

from structures import Point, Size, Colour, BLUE, DynamicColour
from structures.factory import gen_set, gen_heap
from structures.heap import Heap

from simulator.services.services import Services

from simulator.views.map.display.gradient_list_map_display import GradientListMapDisplay
from simulator.views.map.display.map_display import MapDisplay
from simulator.views.map.display.solid_iterable_map_display import SolidIterableMapDisplay

'''
        
    def load_maps(self) -> List[Map]:
        maps = list()
        for i in range(50):
            maps.append("testing_maps_pickles/block_map_1000/" + str(i))
            maps.append("testing_maps_pickles/house_1000/" + str(i))
            maps.append("testing_maps_pickles/uniform_random_fill_1000/" + str(i))
        return maps
    
    def __get_a_star_results(self, maps:List[Map]) -> List[Dict[str, Any]]:
        results = list()
        for _, grid in enumerate(maps):
            result = self.__run_simulation(grid, AStar, AStarTesting, ([], {}))
            results.append(result)

        # synthesis_results = self.__get_results(results)

        return results
    
    def __get_improvement_result(self, res_proc, a_star_res_proc):
        def __get_improvement(val, against, low_is_better=True):
            if against == 0:
                return 0
            res = float(val - against) / float(against) * 100
            res = round(res if not low_is_better else -res, 2)
            return res
        
        goal_found_perc_improvement = __get_improvement(res_proc["goal_found_perc"], a_star_res_proc["goal_found_perc"], False)
        average_steps_improvement = __get_improvement(res_proc["average_steps"], a_star_res_proc["average_steps"])
        average_distance_improvement = __get_improvement(res_proc["average_distance"], a_star_res_proc["average_distance"])
        average_smoothness_improvement = __get_improvement(res_proc["average_smoothness"], a_star_res_proc["average_smoothness"])
        average_clearance_improvement = __get_improvement(res_proc["average_clearance"], a_star_res_proc["average_clearance"], False)
        average_time_improvement = __get_improvement(res_proc["average_time"], a_star_res_proc["average_time"])
        average_distance_from_goal_improvement = __get_improvement(res_proc["average_distance_from_goal"], a_star_res_proc["average_distance_from_goal"])
        average_memory_improvement = __get_improvement(res_proc["average memory"], a_star_res_proc["average memory"])

        res_proc["goal_found_perc_improvement"] = goal_found_perc_improvement
        res_proc["average_steps_improvement"] = average_steps_improvement
        res_proc["average_distance_improvement"] = average_distance_improvement
        res_proc["average_smoothness_improvement"] = average_smoothness_improvement
        res_proc["average_clearance_improvement"] = average_clearance_improvement
        res_proc["average_time_improvement"] = average_time_improvement
        res_proc["average_distance_from_goal_improvement"] = average_distance_from_goal_improvement
        res_proc["average_path_deviation"] = res_proc["average_distance"] - a_star_res_proc["average_distance"]
        res_proc['path_deviation_alldata'] = [y-x for x, y in zip(res_proc['path_deviation_alldata'], a_star_res_proc["distance_alldata"])]
        res_proc['average_memory_improvement'] = average_memory_improvement
    
    @staticmethod
    def __get_average_value(results: List[Dict[str, Any]], attribute: str, decimal_places: int = 2) -> float:
        if len(results) == 0:
            return 0
        # list of data for each result type
        # lstdata =list(map(lambda r: r[attribute], results)))
        val: float = sum(list(map(lambda r: r[attribute], results)))
        val = round(float(val) / len(results), decimal_places)
        return val

    @staticmethod
    def __get_values(results: List[Dict[str, Any]], attribute: str, decimal_places: int = 2) -> float:
        if len(results) == 0:
            return 0

        # list of data for each result type
        lstdata = list(map(lambda r: r[attribute], results))

        return lstdata
    
    @staticmethod
    def __get_results(results: List[Dict[str, Any]], custom_pick: List[bool] = None) -> Dict[str, Any]:
        goal_found: float = round(PATHPLANNING.__get_average_value(results, "goal_found", 4) * 100, 2)

        if custom_pick:
            filtered_results = list(map(lambda el: el[0], filter(lambda el: el[1], zip(results, custom_pick))))
        else:
            filtered_results = list(filter(lambda r: r["goal_found"], results))

        average_steps: float = PATHPLANNING.__get_average_value(filtered_results, "total_steps")
        average_distance: float = PATHPLANNING.__get_average_value(filtered_results, "total_distance")
        average_smoothness: float = PATHPLANNING.__get_average_value(filtered_results, "smoothness_of_trajectory")
        average_clearance: float = PATHPLANNING.__get_average_value(filtered_results, "obstacle_clearance")
        average_time: float = PATHPLANNING.__get_average_value(filtered_results, "total_time", 4)
        average_distance_from_goal: float = PATHPLANNING.__get_average_value(results, "distance_to_goal")
        average_original_distance_from_goal: float = PATHPLANNING.__get_average_value(results, "original_distance_to_goal")

        steps_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "total_steps")
        distance_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "total_distance")
        smoothness_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "smoothness_of_trajectory")
        clearance_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "obstacle_clearance")
        time_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "total_time", 4)
        distance_from_goal_alldata: List[Any] = PATHPLANNING.__get_values(results, "distance_to_goal")
        original_distance_from_goal_alldata: List[Any] = PATHPLANNING.__get_values(results, "original_distance_to_goal")
        path_deviation_alldata: List[Any] = PATHPLANNING.__get_values(filtered_results, "total_distance")
        memory_alldata = PATHPLANNING.__get_values(filtered_results, "memory")
        average_memory = PATHPLANNING.__get_average_value(results, "memory")

        ret = {
            "goal_found_perc": goal_found,
            "average_steps": average_steps,
            "average_distance": average_distance,
            "average_smoothness": average_smoothness,
            "average_clearance": average_clearance,
            "average_time": average_time,
            "average_distance_from_goal": average_distance_from_goal,
            "average_original_distance_from_goal": average_original_distance_from_goal,
            "steps_alldata": steps_alldata,
            "distance_alldata": distance_alldata,
            "smoothness_alldata": smoothness_alldata,
            "clearance_alldata": clearance_alldata,
            "time_alldata": time_alldata,
            "distance_from_goal_alldata": distance_from_goal_alldata,
            "original_distance_from_goal_alldata": original_distance_from_goal_alldata,
            'path_deviation_alldata': distance_alldata,
            'memory_alldata': memory_alldata,
            'average memory': average_memory
        }

        return ret

    
    def __run_simulation(self, grid: Map, algorithm_type: Type[Algorithm], testing_type: Type[BasicTesting],
                        algo_params: Tuple[list, dict], agent_pos: Point = None) -> Dict[str, Any]:
        config = Configuration()
        config.simulator_initial_map = copy.deepcopy(grid)
        config.simulator_algorithm_type = algorithm_type
        config.simulator_testing_type = testing_type
        config.simulator_algorithm_parameters = algo_params

        if agent_pos:
            config.simulator_initial_map.move_agent(agent_pos, True, False)

        sim: Simulator = Simulator(Services(config))

        tracemalloc.start()

        resu = sim.start().get_results()

        _, peak = tracemalloc.get_traced_memory()  # current, peak

        tracemalloc.stop()

        resu['memory'] = (peak/1000)

        return resu
    
    def __evaluate_path(self, alg) -> float:
        planning_module = alg.PathPlanning
        testing_type = BasicTesting
        algo_params = ([], {})
        
        results: List[Dict[str, Any]] = list()
        fail_count = 0
        time_limit_count = 0
        for _, grid in enumerate(self.maps):
            results.append(self.__run_simulation(grid, planning_module, testing_type, algo_params))
            fail_count += not results[-1].get("goal_found", False)
            time_limit_count += results[-1].get("total_time", 0.0) > 10.0

            if fail_count > len(self.maps)//10: # Early stop of bad algorithm evaluation
                return float("inf"), {}
            elif time_limit_count > len(self.maps)//5:
                return float("inf"), {}
            
        res_proc = self.__get_results(results)

        a_star_res = self.a_star_results
        a_star_res_proc = self.__get_results(a_star_res, list(map(lambda r: r["goal_found"], results)))

        self.__get_improvement_result(res_proc, a_star_res_proc)

        fitness = 100 * res_proc["goal_found_perc_improvement"] + \
                    0.5 * res_proc["average_distance_improvement"] + \
                    0 * res_proc["average_time_improvement"] + \
                    res_proc["average_smoothness_improvement"] + \
                    3 * res_proc["average_clearance_improvement"] + \
                    2 * res_proc["average_memory_improvement"]

        return -fitness, res_proc

        
    def evaluate(self, code_string):
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

                fitness, results = self.__evaluate_path(planning_module)

                return fitness, results
        
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return None, {"Traceback" : traceback.format_exc()}
        

    def get_import_string(self):
        return self.import_string



