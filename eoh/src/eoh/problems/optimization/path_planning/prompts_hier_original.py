from eoh.methods.eoh.classic_planning_method import GetPlanningCode

class GetPrompts():
    def __init__(self):
        self.planning_code = GetPlanningCode()
        
        self.role = "You are given a reference implementations for path planning algorithms on a discrete grid environment."
        
        self.prompt_task = "Your task is to design and implement an **improved path planning algorithm**, written as a Python function named `_find_path_internal`, that is inspired by but not limited to the provided examples."

        self.predefined_information = '''

'''

        self.objective = '''
### Objective:
- Improve path planning performance in terms of:
  - Planning efficiency (e.g., fewer iterations)
  - Path quality (e.g., smoother, shorter)
  - Robustness (e.g., fewer failures to connect start and goal)
  - Success rate (e.g., more successful connections)
- You may use techniques like:
    - Goal-biased or adaptive sampling
    - Heuristic-guided expansion (e.g., A* cost)
    - Adaptive step size (`max_dist`) based on environment
    - Rewiring or optimization steps (e.g., RRT*)
    - Smoothed or shortcut path extraction
    - Early stopping criteria or dynamic iteration limits
    ... and more.
'''
        self.constraints = '''
### Constraints:
- Please write a brief description of the algorithm you generated.
- The description must be inside a brace and placed at the very top of the code.
- Implement it in Python.
- You do not need to declare the imports, as they are already provided in the codebase.
- Your function must be named `_find_path_internal`.
- You must reuse existing helper functions where applicable. If necessary, you may define and use new helper functions to support the implementation.
- It should work with existing components: `Forest`, `Point`, `Vertex`, etc.
- The `__init__` method must not be modified. However, you are allowed to add new member variables within it (no structural changes or logic modifications).
- When referencing multiple algorithms, don't forget to declare variables in __init__.
- The core logic of the path planning algorithm must be implemented inside the `_find_path_internal` function. You may call any helper functions from within `_find_path_internal`.
- Analyze the usage patterns and conventions from the provided codebase (e.g., class structure, function calls, and service access), and ensure your code follows the same patterns.
- All variables or objects used in the code must be explicitly declared before use. Do not use undeclared variables, even if they appear to be implied from context.
- If the reference code uses specific variable declarations (e.g., `self._graph`, `self._q_new`, 'self._get_random_sample', etc.), ensure these are preserved and correctly initialized before being used.
- Always verify that any newly introduced variables are properly initialized and assigned in a contextually valid location.
- Do not assume the existence of any variables that are not shown in the provided reference code. If a variable is required, define it explicitly and ensure it is logically scoped.
- After code generation, you must review the code to ensure it is syntactically correct, logically coherent, and executable within the expected environment.
- Add code to treat a route search as not found if it takes more than 30 seconds to find the route.(in function `_find_path_internal`)

### You may freely define new helper functions if necessary
- If your approach benefits from additional utility methods (e.g., cost estimation, region sampling, custom distance functions), feel free to create and use them.

### The `_find_path_internal` function is the main function executed for path planning.
'''

        self.prompt_func_name = ""
        self.prompt_func_inputs = ""
        self.prompt_func_outputs = ""


        self.prompt_inout_inf = ""
        self.prompt_other_inf = "A Python class implementing an improved path planner named `PathPlanning`."

        self.package_info = '''
from structures import Point
from algorithms.configuration.maps.map import Map
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest


### Additional Reference Information

`Map` class:

Map is a class that represents a grid-based environment for path planning. 
It provides methods to manage the grid, including checking valid positions, moving the agent, and retrieving line sequences. 
The class also supports various operations related to the grid's size, obstacles, and agent's position.
This class represents the environment in which the agent can move with dimensions corresponding to the agent.
It contains 1 agent, 1 goal and multiple obstacles.

Constructor: Map(agent: Agent, goal: Goal, obstacles: List[Obstacle])

Attributes:
- agent: Agent
- goal: Goal
- obstacles: List[Obstacle]
- trace: List[Trace]
- size: Size

Agent, Goal, Obstacle and Trace are classes that represent the agent, goal, obstacle and trace positions in the grid. Access to their attributes positions and radius.
(e.g. `.position`, `.radius`)

methods:
- get_obstacle_bound(self, obstacle_start_point: Point, visited: Optional[Set[Point]] = None) -> Set[Point]: Method for finding the bound of an obstacle by doing a fill
:param obstacle_start_point: The obstacle location
:param visited: Can pass own visited set so that we do not visit those nodes again
:return: Obstacle bound

- get_move_index(self, direction: List[float]) -> int: Method for getting the move index from a direction
:param direction: The direction
:return: The index as an integer

- get_move_along_dir(self, direction: List[float]) -> Point: Method for getting the movement direction from a normal direction
:param direction: The true direction
:return: The movement direction as a Point

- reset(self) -> None: Resets the map by replaying the trace

- get_line_sequence(self, frm: Point, to: Point) -> List[Point]: Bresenham's line algorithm, Given coordinate of two n dimensional points. The task to find all the intermediate points required for drawing line AB.

- is_valid_line_sequence(self, line_sequence: List[Point]) -> bool

- is_goal_reached(self, pos: Point, goal: Goal = None) -> bool: Method that checks if the position coincides with the goal
:param pos: The position
:param goal: If not default goal is considered
:return: If the goal has been reached from the given position

- is_agent_in_goal_radius(self, agent_pos: Point = None, goal: Goal = None) -> bool

- is_agent_valid_pos(self, pos: Point) -> bool: Checks if the point is a valid position for the agent
:param pos: The position
:return: If the point is a valid position for the agent

- get_next_positions(self, pos: Point) -> List[Point]: Returns the next available positions valid from the agent point of view given x-point connectivity
:param pos: The position
:return: A list of positions

- get_next_positions_with_move_index(self, pos: Point) -> List[Tuple[Point, int]]: Returns the next available positions valid from the agent point of view given x-point connectivity
:param pos: The position
:return: A list of positions with move index

- apply_move(move: Point, pos: Point) -> Point: Applies the given move and returns the new destination (this is staticmethod function, it can use like this `Map.apply_move(move, pos)` or `self._get_grid().apply_move(move, pos)`)
:param move: The move to apply
:param pos: The source
:return: The destination after applying the move

- get_movement_cost(self, frm: Union[Point, Entity] = None, to: Union[Point, Entity] = None) -> float: Returns the movement cost from one point to another
:param frm: The source point or entity
:param to: The destination point or entity
:return: The movement cost as a float

- get_movement_cost_from_index(self, idx: int, frm: Optional[Point] = None) -> float: Returns the movement cost from one point to another using the move index

- get_distance(frm: Point, to: Point) -> float: Returns the distance between two points (this is staticmethod function, it can use like this `Map.get_distance(frm, to)` or `self._get_grid().get_distance(frm, to)`)



`Vertex` class:
The `Vertex` class represents a node in the path planning graph. It stores the position of the vertex, its children and parents, and the cost associated with it. The class provides methods to add child and parent vertices, allowing for the construction of a directed graph structure.

Constructor: Vertex(position: Point, store_connectivity: bool = False)

Attributes:
- position: Point
- cost: float
- children: Set['Vertex']
- parents: Set['Vertex']
- connectivity: Set['Vertex']
- aux: Dict[Any, Any]

This Attributes cans access the vertex position, cost, children, parents, connectivity and auxiliary data.
(e.g. `.position`, `.cost`, `.children`, `.parents`, `.connectivity`, `.aux`)

Methods:

- add_child(self, child: 'Vertex') -> None: Adds a child vertex to the current vertex.
- add_parent(self, parent: 'Vertex') -> None: Adds a parent vertex to the current vertex.
- set_child(self, child: 'Vertex'): Sets a child vertex for the current vertex.
- set_parent(self, parent: 'Vertex'): Sets a parent vertex for the current vertex.
- visit_children(self, f: Callable[['Vertex'], bool]) -> None: Visits all child vertices and applies a function to them.
- visit_parents(self, f: Callable[['Vertex'], bool]) -> None: Visits all parent vertices and applies a function to them.


`Forest` class:
The `Forest` class represents a collection of trees (graphs) in the path planning algorithm. It allows for the management of multiple root vertices and provides methods to add edges, find nearest vertices, and retrieve vertices within a certain radius.

Attributes:
- root_vertex_start: Vertex
- root_vertex_goal: Vertex
- root_vertices: List[Vertex]
- size: int

This Attributes cans access the root vertices, size of the forest and root vertices for start and goal.
(e.g. `.root_vertex_start`, `.root_vertex_goal`, `.root_vertices`, `.size`)

Methods:
- reverse_root_vertices(self) -> None: Reverses the order of root vertices in the forest.
- get_random_vertex(self, root_vertices: List[Vertex]) -> Vertex: Returns a random vertex from the specified root vertices.
- get_nearest_vertex(self, root_vertices: List[Vertex], point: Point) -> Vertex: Returns the nearest vertex to a given point from the specified root vertices.
- get_vertices_within_radius(self, root_vertices: List[Vertex], point: Point, radius: float) -> List[Vertex]: Returns a list of vertices within a specified radius from a given point in the specified root vertices.
- add_edge(self, parent: Vertex, child: Vertex): Adds an edge between two vertices in the forest.
- remove_edge(self, parent: Vertex, child: Vertex): Removes an edge between two vertices in the forest.
- walk_dfs_subset_of_vertices(self, root_vertices_subset: List[Vertex], f: Callable[[Vertex], bool]): Performs DFS from each vertex in the subset, applying f to all descendants.
- walk_dfs(self, f: Callable[[Vertex], bool]): Performs DFS from all root vertices, applying f to all descendants.

`CyclicGraph` class:
The `CyclicGraph` class is a specialized version of the `Forest` class that allows for the addition of edges between vertices, enabling cyclic connections. It inherits from the `Forest` class and provides additional functionality for managing cyclic relationships between vertices.

Attributes:
- root_vertex_start: Vertex
- root_vertex_goal: Vertex
- root_vertices: List[Vertex]
- size: int

This Attributes cans access the root vertices, size of the forest and root vertices for start and goal.
(e.g. `.root_vertex_start`, `.root_vertex_goal`, `.root_vertices`, `.size`)

Methods:
- reverse_root_vertices(self) -> None: Reverses the order of root vertices in the forest.
- get_random_vertex(self, root_vertices: List[Vertex]) -> Vertex: Returns a random vertex from the specified root vertices.
- get_nearest_vertex(self, root_vertices: List[Vertex], point: Point) -> Vertex: Returns the nearest vertex to a given point from the specified root vertices.
- get_vertices_within_radius(self, root_vertices: List[Vertex], point: Point, radius: float) -> List[Vertex]: Returns a list of vertices within a specified radius from a given point in the specified root vertices.
- walk_dfs_subset_of_vertices(self, root_vertices_subset: List[Vertex], f: Callable[[Vertex], bool]): Applies f to each vertex in the subset; stops early if f returns False.
- walk_dfs(self, f: Callable[[Vertex], bool]): Applies f to each root vertex; stops early if f returns False.




declare intial variables:
self.grid: Map = self._get_grid()

start_vertex: Vertex = Vertex(self.grid.agent.position)
start_vertex.cost = 0
goal_vertex: Vertex = Vertex(self.grid.goal.position)

self._graph: Forest = gen_forest(self._services, start_vertex, goal_vertex, [])


### Additional Reference Information (Map API Access)
You may use the following `Map` methods and properties to support path planning:
- `self._get_grid().agent`: the agent's current position (`Entity`)
    - `self.grid.agent.get_position()`: returns the agent's position as a `Point`
- `self._get_grid().goal`: the goal's position (`Entity`)
    - `self._get_grid().goal.get_position()`: returns the goal's position as a `Point`
- `self._get_grid().obstacles`: a list of obstacles (`List[Obstacle]`)

'''

        self.combined_sample_based_algorithm = '''
`SampleBasedAlgorithm` class:

Methods:
- _get_grid(self) -> Map: Gets the Map object
- move_agent(self, to: Point) -> None: Moves the agent to the specified point
- key_frame(self, *args, **kwargs) -> None: this function must add a key frame to the algorithm's execution, used for debugging and visualization purposes.
'''


# TODO: Add the helper functions to the prompt
        self.helper_funtion = '''
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

    
'''
    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
    
    def get_role(self):
        return self.role
    def get_objective(self):
        return self.objective
    def get_constraints(self):
        return self.constraints

    def get_prior_knowledge(self):
        return
    
# prompt
'''
start_vertex = Vertex(self._get_grid().agent.position)
start_vertex.cost = 0
goal_vertex = Vertex(self._get_grid().goal.position)

self._graph = gen_forest(self._services, start_vertex, goal_vertex, [])


'''