class GetPlanningCode():
    def __init__(self):
        self.a_star = '''
from typing import Set, List, Tuple, Optional, Dict

import numpy as np


from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.maps.map import Map
from algorithms.configuration.maps.ros_map import RosMap
from simulator.services.services import Services
from simulator.views.map.display.gradient_list_map_display import GradientListMapDisplay
from simulator.views.map.display.map_display import MapDisplay
from simulator.views.map.display.solid_iterable_map_display import SolidIterableMapDisplay
from structures import Point, Colour, BLUE, DynamicColour
from structures.factory import gen_set, gen_heap
from structures.heap import Heap

from memory_profiler import profile

class AStar(Algorithm):
    class InternalMemory:
        priority_queue: Heap
        visited: Set[Point]
        back_pointer: Dict[Point, Optional[Point]]
        g: Dict[Point, int]
        h: Dict[Point, float]

        def __init__(self, services: Services):
            self.priority_queue = gen_heap(services)
            self.visited = gen_set(services)
            self.back_pointer = {}
            self.g = {}
            self.h = {}

    mem: InternalMemory

    pq_colour_max: DynamicColour
    pq_colour_min: DynamicColour
    visited_colour: DynamicColour

    __map_displays: List[MapDisplay]

    def __init__(self, services: Services, testing: BasicTesting = None):
        super().__init__(services, testing)

        self.mem = AStar.InternalMemory(self._services)

        self.pq_colour_max = self._services.state.views.add_colour("explored max", BLUE)
        self.pq_colour_min = self._services.state.views.add_colour("explored min", Colour(0.27, 0.33, 0.35, 0.2))
        self.visited_colour = self._services.state.views.add_colour("visited", Colour(0.19, 0.19, 0.2, 0.8))

        self.__map_displays = [SolidIterableMapDisplay(self._services, self.mem.visited, self.visited_colour, z_index=50),
                               GradientListMapDisplay(self._services, self.mem.priority_queue, min_colour=self.pq_colour_min,
                                                  max_colour=self.pq_colour_max, z_index=49, inverted=True)]

    def set_display_info(self) -> List[MapDisplay]:
        """
        Read super description
        """
        return super().set_display_info() + self.__map_displays

    # noinspection PyUnusedLocal
    # @profile
    def _find_path_internal(self) -> None:
        self._init_mem()

        if self._expand():
            self._follow_back_trace()

    def _init_mem(self) -> None:
        grid: Map = self._get_grid()

        # push agent
        self.mem.g[grid.agent.position] = 0
        item: Tuple[float, Point] = (self.get_heuristic(grid.agent.position), grid.agent.position)
        self.mem.priority_queue.push(item)
        self.mem.back_pointer[grid.agent.position] = None

    def _expand(self) -> bool:
        grid: Map = self._get_grid()

        while len(self.mem.priority_queue) > 0:
            total_dist: float
            next_node: Point
            # peek and check if we need to terminate
            total_dist, next_node = self.mem.priority_queue.pop()

            if grid.is_goal_reached(next_node):
                self.mem.priority_queue.push((total_dist, next_node))
                return True

            self.mem.visited.add(next_node)

            for n, idx in grid.get_next_positions_with_move_index(next_node):
                if n not in self.mem.visited:
                    dist = grid.get_movement_cost_from_index(idx, n)
                    if n not in self.mem.g or self.mem.g[next_node] + dist < self.mem.g[n]:
                        # it does not matter if we have duplicates as we will not be looking at them
                        # therefore it does not affect the priority
                        self.mem.g[n] = self.mem.g[next_node] + dist
                        item = (self.f(n), n)
                        self.mem.priority_queue.push(item)
                        self.mem.back_pointer[n] = next_node

            self.key_frame()
        return False

    def f(self, x: Point) -> float:
        g = self.mem.g[x]
        h = self.get_heuristic(x)
        ret = g + h
        return ret

    def _follow_back_trace(self):
        grid: Map = self._get_grid()
        
        trace: List[Point] = self.get_back_trace(grid.goal)
        trace.reverse()
        for t in trace:
            self.move_agent(t)
            if isinstance(grid, RosMap):
                grid.publish_wp(grid.agent.position)
            self.key_frame(ignore_key_frame_skip=True)

    def get_back_trace(self, goal: Goal) -> List[Point]:
        """
        Follows the back pointer until it gets to the agent position
        :return: The trace
        """
        trace = []
        pos = goal.position
        while self.mem.back_pointer[pos] is not None:
            trace.append(pos)
            pos = self.mem.back_pointer[pos]
        return trace

    def get_heuristic(self, pos: Point) -> float:
        """
        Returns the euclidean distance from the given position to the goal
        It does memoization as well
        :param goal: The goal
        :param pos: The initial position
        :return:
        """
        self.mem.h.setdefault(pos, np.linalg.norm(np.array(pos) - np.array(self._get_grid().goal.position)))
        return self.mem.h[pos]
'''
        self.rrt_connect = '''
from typing import List

import torch
import numpy as np

from memory_profiler import profile

from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.basic_testing import BasicTesting
from simulator.services.services import Services
from structures import Point

from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest

n=0

class RRT_Connect(SampleBasedAlgorithm):
    _graph: Forest
    _max_dist: float
    _iterations: int

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        super().__init__(services, testing)
        
        self._graph = gen_forest(self._services, Vertex(self._get_grid().agent.position), Vertex(self._get_grid().goal.position), [])
        self._graph.edges_removable = False
        self._init_displays()
        
        self._max_dist = 10
        self._iterations = 10000

    # Helper Functions #
    # -----------------#

    def _extend(self, root_vertex: Vertex, q: Point) -> str:
        self._q_near: Vertex = self._get_nearest_vertex(root_vertex, q)
        self._q_new: Vertex = self._get_new_vertex(self._q_near, q, self._max_dist)
        if self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(self._q_near.position, self._q_new.position)):
            self._graph.add_edge(self._q_near, self._q_new)
            if self._q_new.position == q:
                return 'reached'
            else:
                return 'advanced'
        return 'trapped'

    def _connect(self, root_vertex: Vertex, q: Vertex) -> str:
        S = 'advanced'
        while S == 'advanced':
            S = self._extend(root_vertex, q.position)
        self._mid_vertex = q
        return S

    def _extract_path(self):

        # trace back
        path_mid_to_b: List[Vertex] = [self._q_new]

        while len(path_mid_to_b[-1].parents) != 0:
            for parent in path_mid_to_b[-1].parents:
                path_mid_to_b.append(parent)
                break

        path_a_to_mid: List[Vertex] = [self._extension_target]

        while len(path_a_to_mid[-1].parents) != 0:
            for parent in path_a_to_mid[-1].parents:
                path_a_to_mid.append(parent)
                break

        path_a_to_mid.reverse()
        path = path_a_to_mid + path_mid_to_b

        if self._graph.root_vertices[0] is self._graph.root_vertex_goal:
            path.reverse()

        for p in path:
            self.move_agent(p.position)
            self.key_frame(ignore_key_frame_skip=True)

    def _get_random_sample(self) -> Point:
        while True:
            rand_pos = np.random.randint(0, self._get_grid().size, self._get_grid().size.n_dim)
            sample: Point = Point(*rand_pos)
            if self._get_grid().is_agent_valid_pos(sample):
                return sample

    def _get_nearest_vertex(self, graph_root_vertex: Vertex, q_sample: Point) -> Vertex:
        return self._graph.get_nearest_vertex([graph_root_vertex], q_sample)

    def _get_new_vertex(self, q_near: Vertex, q_sample: Point, max_dist) -> Vertex:
        dir = q_sample.to_tensor() - q_near.position.to_tensor()
        if torch.norm(dir) <= max_dist:
            return Vertex(q_sample)

        dir_normalized = dir / torch.norm(dir)
        q_new = Point.from_tensor(q_near.position.to_tensor() + max_dist * dir_normalized)
        return Vertex(q_new)

    # Overridden Implementation #
    # --------------------------#

    #@profile
    def _find_path_internal(self) -> None:

        for i in range(self._iterations):
            global n 
            n+=1

            if n == 100:
                break

            q_rand: Point = self._get_random_sample()

            if not self._extend(self._graph.root_vertices[0], q_rand) == 'trapped':
                self._extension_target = self._q_new
                if self._connect(self._graph.root_vertices[-1], self._q_new) == 'reached':
                    self._extract_path()
                    break
            self._graph.reverse_root_vertices()

            # visualization code
            self.key_frame()
'''

        self.rrt_star = '''
from typing import List

import torch
import numpy as np

from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.basic_testing import BasicTesting
from simulator.services.services import Services
from structures import Point
from algorithms.configuration.maps.ros_map import RosMap
from algorithms.configuration.maps.map import Map
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest


class RRT_Star(SampleBasedAlgorithm):
    _graph: Forest

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        super().__init__(services, testing)
        
        start_vertex = Vertex(self._get_grid().agent.position)
        start_vertex.cost = 0
        goal_vertex = Vertex(self._get_grid().goal.position)

        self._graph = gen_forest(self._services, start_vertex, goal_vertex, [])
        self._init_displays()

    # Helper Functions #
    # -----------------#

    def _get_random_sample(self) -> Point:
        while True:
            rand_pos = np.random.randint(0, self._get_grid().size, self._get_grid().size.n_dim)
            sample: Point = Point(*rand_pos)
            if self._get_grid().is_agent_valid_pos(sample):
                return sample

    def _get_nearest_vertex(self, q_sample: Point) -> Vertex:
        return self._graph.get_nearest_vertex([self._graph.root_vertex_start], q_sample)

    def _get_vertices_within_radius(self, vertex: Vertex, radius: float) -> List[Vertex]:
        return self._graph.get_vertices_within_radius([self._graph.root_vertex_start], vertex.position, radius)

    def _get_new_vertex(self, q_near: Vertex, q_sample: Point, max_dist) -> Vertex:
        dir = q_sample.to_tensor() - q_near.position.to_tensor()
        dir_norm = torch.norm(dir)
        if dir_norm <= max_dist:
            return Vertex(q_sample)

        dir_normalized = dir / torch.norm(dir)
        q_new = Point.from_tensor(q_near.position.to_tensor() + max_dist * dir_normalized)
        return Vertex(q_new)

    def _extract_path(self, q_new):

        goal_v: Vertex = Vertex(self._get_grid().goal.position)
        child_parent_dist = torch.norm(q_new.position.to_tensor() - goal_v.position.to_tensor())
        goal_v.cost = q_new.cost + child_parent_dist
        self._graph.add_edge(q_new, goal_v)
        path: List[Vertex] = [goal_v]

        while len(path[-1].parents) != 0:
            for parent in path[-1].parents:
                path.append(parent)
                break

        del path[-1]
        path.reverse()

        for p in path:
            self.move_agent(p.position)
            #sends waypoint for ros extension
            grid: Map = self._get_grid()
            if isinstance(grid, RosMap):
                grid.publish_wp(grid.agent.position)
            self.key_frame(ignore_key_frame_skip=True)

    # Overridden Implementation #
    # --------------------------#

    def _find_path_internal(self) -> None:

        max_dist: float = 10
        iterations: int = 10000
        max_radius: float = 50
        lambda_rrt_star: float = 50
        dimension = 2

        for i in range(iterations):

            q_sample: Point = self._get_random_sample()
            q_nearest: Vertex = self._get_nearest_vertex(q_sample)
            if q_nearest.position == q_sample:
                continue
            q_new: Vertex = self._get_new_vertex(q_nearest, q_sample, max_dist)

            if not self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(q_nearest.position, q_new.position)):
                continue

            card_v = torch.tensor(float(self._graph.size))
            log_card_v = torch.log(card_v)
            radius = min(lambda_rrt_star*((log_card_v/card_v)**(1/dimension)),max_radius)
            Q_near: List[Vertex] = self._get_vertices_within_radius(q_new, radius)
            q_min = q_nearest
            c_min = q_nearest.cost + torch.norm(q_nearest.position.to_tensor() - q_new.position.to_tensor())

            for q_near in Q_near:
                near_new_collision_free = self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(q_near.position, q_new.position))
                cost_near_to_new = q_near.cost + torch.norm(q_near.position.to_tensor() - q_new.position.to_tensor())
                if near_new_collision_free and cost_near_to_new < c_min:
                    q_min = q_near
                    c_min = cost_near_to_new

            child_parent_dist = torch.norm(q_min.position.to_tensor() - q_new.position.to_tensor())
            q_new.cost = q_min.cost + child_parent_dist
            self._graph.add_edge(q_min, q_new)

            for q_near in Q_near:
                near_new_collision_free = self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(q_near.position, q_new.position))
                cost_new_to_near = q_new.cost + torch.norm(q_new.position.to_tensor() - q_near.position.to_tensor())
                if near_new_collision_free and cost_new_to_near < q_near.cost:
                    q_parent = None
                    for parent in q_near.parents:
                        q_parent = parent
                        break
                    q_near.cost = None
                    self._graph.remove_edge(q_parent, q_near)
                    child_parent_dist = torch.norm(q_new.position.to_tensor() - q_near.position.to_tensor())
                    q_near.cost = q_new.cost + child_parent_dist
                    self._graph.add_edge(q_new, q_near)

            if self._get_grid().is_agent_in_goal_radius(agent_pos=q_new.position):
                self._extract_path(q_new)
                break

            self.key_frame()

'''
        self.rrt='''
from typing import List

import torch
import numpy as np

from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.basic_testing import BasicTesting
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest

from simulator.services.services import Services

from structures import Point


class RRT(SampleBasedAlgorithm):
    _graph: Forest

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        super().__init__(services, testing)
        
        self._graph = gen_forest(self._services, Vertex(self._get_grid().agent.position), Vertex(self._get_grid().goal.position), [])
        self._graph.edges_removable = False
        self._init_displays()

    # Helper Functions #
    # -----------------#

    def _get_new_vertex(self, q_near: Vertex, q_sample: Point, max_dist) -> Vertex:
        dir = q_sample.to_tensor() - q_near.position.to_tensor()
        if torch.norm(dir) <= max_dist:
            return Vertex(q_sample)

        dir_normalized = dir / torch.norm(dir)
        q_new = Point.from_tensor(q_near.position.to_tensor() + max_dist * dir_normalized)
        return Vertex(q_new)

    def _get_random_sample(self) -> Point:
        while True:
            rand_pos = np.random.randint(0, self._get_grid().size, self._get_grid().size.n_dim)
            sample: Point = Point(*rand_pos)
            if self._get_grid().is_agent_valid_pos(sample):
                return sample

    def _extract_path(self, q_new):

        goal_v: Vertex = Vertex(self._get_grid().goal.position)
        self._graph.add_edge(q_new, goal_v)    #connect the last sampled point that's close to goal vertex and connet point to goal vertex with edge
        path: List[Vertex] = [goal_v]    

        while len(path[-1].parents) != 0:
            for parent in path[-1].parents:
                path.append(parent)
                break

        del path[-1]
        path.reverse()

        #get animation of path tracing from start to goal
        for p in path:
            self.move_agent(p.position)   
            self.key_frame(ignore_key_frame_skip=True)

    # Overridden Implementation #
    # --------------------------#

    def _find_path_internal(self) -> None:

        max_dist: float = 10
        iterations: int = 10000

        for i in range(iterations):

            q_sample: Point = self._get_random_sample()     #sample a random point and return it if it's in valid position
            q_near: Vertex = self._graph.get_nearest_vertex([self._graph.root_vertex_start], q_sample) 
            if q_near.position == q_sample:
                continue    #restart the while loop right away if sample point same as nearest vertex point
            q_new: Vertex = self._get_new_vertex(q_near, q_sample, max_dist)    #get new vertex

            if not self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(q_near.position, q_new.position)):    
                continue    #restart the while loop right away if the straight line path from nearest vertex to new sample point is invalid 
            self._graph.add_edge(q_near, q_new)    #add edge between 2 points

            if self._get_grid().is_agent_in_goal_radius(agent_pos=q_new.position):    #if agent is in goal radius, then run _extract_path method 
                self._extract_path(q_new)
                break

            self.key_frame()    #add the new vertex and edge if the new sample point is not at goal yet
'''
        self.sprm = '''
from typing import List

import torch

from algorithms.classic.sample_based.core.sample_based_algorithm import SampleBasedAlgorithm
from algorithms.basic_testing import BasicTesting
from simulator.services.services import Services
from structures import Point

from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_cyclic_graph, CyclicGraph


class SPRM(SampleBasedAlgorithm):
    _graph: CyclicGraph
    _V_size: int
    _max_radius: float

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        super().__init__(services, testing)
        self._V_size = 200
        self._max_radius = 15
        V: List[Vertex] = list()
        for i in range(self._V_size):
            q_rand: Point = self._get_random_sample()
            V.append(Vertex(q_rand, store_connectivity=True))
        self._graph = gen_cyclic_graph(self._services,
                                       Vertex(self._get_grid().agent.position, store_connectivity=True),
                                       Vertex(self._get_grid().goal.position, store_connectivity=True),
                                       V)
        self._graph.edges_removable = False
        self._init_displays()

    # Helper Functions #
    # -----------------#

    def _near(self, vertex: Vertex) -> List[Vertex]:
        return self._graph.get_vertices_within_radius(self._graph.root_vertices, vertex.position, self._max_radius)

    def _get_random_sample(self) -> Point:
        while True:
            sample = Point(*[torch.randint(0, self._get_grid().size[i], (1,)).item() for i in range(self._get_grid().size.n_dim)])
            if self._get_grid().is_agent_valid_pos(sample):
                return sample

    def _get_new_vertex(self, q_near: Vertex, q_sample: Point, max_dist) -> Vertex:
        dir = q_sample.to_tensor() - q_near.position.to_tensor()
        if torch.norm(dir) <= max_dist:
            return Vertex(q_sample)

        dir_normalized = dir / torch.norm(dir)
        q_new = Point.from_tensor(q_near.position.to_tensor() + max_dist * dir_normalized)
        return Vertex(q_new)

    def _extract_path(self):

        goal: Vertex = self._graph.root_vertices[1]
        agent: Vertex = self._graph.root_vertices[0]

        current_vertex = agent
        path = list()
        while current_vertex is not goal:
            current_vertex = current_vertex.connectivity[goal]
            path.append(current_vertex)

        for p in path:
            self.move_agent(p.position)
            self.key_frame(ignore_key_frame_skip=True)

    # Overridden Implementation #
    # --------------------------#

    def _find_path_internal(self) -> None:

        for i, v in enumerate(self._graph.root_vertices):
            U = self._near(v)
            for u in U:
                if self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(u.position, v.position)):
                    if v is not u:
                        self._graph.add_edge(v, u)
                        self._graph.add_edge(u, v)
                    self.key_frame()
                    if self._graph.root_vertices[1] in self._graph.root_vertices[0].connectivity:
                        self._extract_path()
                        return
'''

        self.algorithms ='''
from utility.threading import Condition
from typing import Optional, List
from abc import ABC, abstractmethod

from algorithms.basic_testing import BasicTesting
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.maps.map import Map
from simulator.services.services import Services
from simulator.views.map.display.entities_map_display import EntitiesMapDisplay
from simulator.views.map.display.map_display import MapDisplay
from simulator.views.map.display.solid_iterable_map_display import SolidIterableMapDisplay
from structures import Point


class Algorithm(ABC):
    """
    Class for defining basic API for algorithms.
    All algorithms must inherit from this class.
    """
    testing: Optional[BasicTesting]
    _services: Services

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        self._services = services
        self.testing = testing
        self.__root_key_frame = self._services.algorithm

    def set_condition(self, key_frame_condition: Condition) -> None:
        """
        This method is used to initialise the debugging condition
        :param key_frame_condition: The condition
        """
        if self.testing is not None:
            self.testing.set_condition(key_frame_condition)

    def get_display_info(self) -> List[MapDisplay]:
        """
        Returns the info displays
        :return: A list of info displays
        """
        if self.testing is not None:
            return self.testing.display_info
        return []

    def key_frame(self, *args, **kwargs) -> None:
        """
        Method that marks a key frame
        It is used in animations
        """
        if self.testing is not None:
            self.testing.key_frame(*args, **kwargs, root_key_frame=self.__root_key_frame)

    def set_root_key_frame(self, algo):
        self.__root_key_frame = algo

    def find_path(self) -> None:
        """
        Method for finding a path from agent to goal
        Movement should be done using the map APIs
        """
        if self.testing is not None:
            self.testing.algorithm_start()
        self._find_path_internal()
        if self.testing is not None:
            self.testing.algorithm_done()

    def _get_grid(self) -> Map:
        """
        Shortcut to get the map
        :return: The map
        """
        return self._services.algorithm.map

    def move_agent(self, to: Point) -> None:
        """
        Method used to move the agent on the map
        :param to: the destination
        """
        #method is in map.py. follow param means Instead of teleport, moves in a straight line
        self._get_grid().move_agent(to, follow=True)

    @abstractmethod
    def set_display_info(self) -> List[MapDisplay]:
        """
        Method used for setting the info displays
        All algorithms must override this method
        :return: A list of info displays
        """
        return []

    @abstractmethod
    def _find_path_internal(self) -> None:
        """
        The internal implementation of :ref:`find_path`
        All algorithms must override this method
        """
        pass
'''

        self.sample_based_algorithm = '''
class SampleBasedAlgorithm(Algorithm):
    __map_displays: List[MapDisplay]

    def _init_displays(self) -> None:
        self.__map_displays = [GraphMapDisplay(self._services, self._graph)]

    def set_display_info(self) -> List[MapDisplay]:
        return super().set_display_info() + self.__map_displays
'''

    def get_code(self, algorithm_name: str) -> str:
        if algorithm_name == "astar":
            return self.algorithms + '\n' + self.a_star
        elif algorithm_name == "rrt_connect":
            return self.algorithms + '\n' + self.sample_based_algorithm + '\n' + self.rrt_connect
        elif algorithm_name == "rrt_star":
            return self.algorithms + '\n' + self.sample_based_algorithm + '\n' + self.rrt_star
        elif algorithm_name == "rrt":
            return self.algorithms + '\n' + self.sample_based_algorithm + '\n' + self.rrt
        elif algorithm_name == "sprm":
            return self.algorithms + '\n' + self.sample_based_algorithm + '\n' + self.sprm
        else:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            return None
        
    def get_algorithm_description(self, algorithm_name: str) -> str:
        if algorithm_name == "astar":
            return "This is the A* algorithm, a classic pathfinding algorithm that uses heuristics to find the shortest path from the agent to the goal."
        elif algorithm_name == "rrt_connect":
            return "This is the RRT-Connect algorithm, a sample-based pathfinding algorithm that connects two trees to find a path."
        elif algorithm_name == "rrt_star":
            return "This is the RRT* algorithm, an optimized version of RRT that finds the shortest path by rewiring the tree as it grows."
        elif algorithm_name == "rrt":
            return "This is the RRT algorithm, a sample-based pathfinding algorithm that explores the space by randomly sampling points and connecting them to the nearest vertex."
        elif algorithm_name == "sprm":
            return "This is the SPRM algorithm, a sample-based pathfinding algorithm that uses a cyclic graph to find a path from the agent to the goal."
        else:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            return None
        
    def get_objective(self, algorithm_name: str) -> str:
        if algorithm_name == "astar":
            return 0
        elif algorithm_name == "rrt_connect":
            return 0
        elif algorithm_name == "rrt_star":
            return 0
        elif algorithm_name == "rrt":
            return 0
        elif algorithm_name == "sprm":
            return 0
        else:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            return None
        
    def get_inherit_prompt(self):
        return f'''
The class you generate must inherit from `SampleBasedAlgorithm`, which itself inherits from `Algorithm`.

You must conform to this interface structure and override the required abstract methods, including:
- `set_display_info(self) -> List[MapDisplay]`
- `_find_path_internal(self) -> None`

{self.algorithms}
{self.sample_based_algorithm}
'''