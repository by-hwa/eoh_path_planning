from eoh.methods.eoh.classic_planning_method import GetPlanningCode

class GetPrompts():
    def __init__(self):
        self.planning_code = GetPlanningCode()
        
        self.prompt_task = "Your task is to design and implement an **improved path planning algorithm**, written as a Python function named `_find_path_internal`, that is inspired by but not limited to the provided examples."

        self.objective = '''
### Objective:
- Improve path planning performance in terms of:
  - Planning efficiency
  - Path quality
  - Robustness
  - Success rate
  - Path smoothness
  - Path lengths
  - Reduce search time
'''
        self.constraints = '''
### Constraints:
- A PYTHON CLASS IMPLEMENTING AN IMPROVED PATH PLANNING ALGORITHM NAMED `PathPlanning`.
- Please write a brief description of the algorithm you generated.
- The description must be inside a brace and placed at the very top of the code.
- Implement it in Python.
- You DO NOT NEED to declare the any imports.
- You may need to define new helper functions if necessary.
- The core logic of the path planning algorithm must be implemented inside the `_find_path_internal` function. You may call any helper functions from within `_find_path_internal`.
- The size of the search area varies from map to map.
- Analyze the usage patterns and conventions from the provided codebase (e.g., class structure, function calls, and service access), and ensure your code follows the same patterns.
- Always verify that any newly introduced variables are properly initialized and assigned in a contextually valid location.
- After code generation, you must review the code to ensure it is syntactically correct, logically coherent, and executable within the expected environment.
⚠️ Important: Add logic to treat path search as **FAILED** if it takes more than 60 seconds.

### You may freely define new helper functions if necessary
- If your approach benefits from additional utility methods (e.g., cost estimation, region sampling, custom distance functions), feel free to create and use them.

### The `_find_path_internal` function is the main function executed for path planning.

⚠️ Do not give additional explanations.
'''

        self.hier_constraints = '''
### Constraints:
- A PYTHON CLASS IMPLEMENTING AN IMPROVED PATH PLANNING ALGORITHM NAMED `PathPlanning`.
- Please write a brief description of the algorithm you generated.
- The description must be inside a brace and placed at the very top of the code.
- Implement it in Python.
- You DO NOT NEED to declare the any imports.
- Your function must be named `_find_path_internal`.
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
⚠️ Important: Add logic to treat path search as **FAILED** if it takes more than 10 seconds.

### YOU ONLY IMPLIMENT CLASS NAMED `PathPlanning` within method `__init__` and `_find_paht_internal`. YOU DON NOT NEED TO IMPLEMENT HELPFER FUNCTION(BUT YOU CAN CALL them ASSUMING THEY WILL BE IMPLEMENTED LATER.)
- Within `_find_path_internal`, use clearly named helper functions to delegate subtasks (e.g., sampling, extending, connecting, or path extraction).  
Do not implement helper functions inline. Just call them assuming they will be implemented later.
DO NOT IMPLEMENT ANY PLACEHOLDER FUNCTION

### The `_find_path_internal` function is the main function executed for path planning.

⚠️ Do not give additional explanations.
'''


        self.package_info = '''
from structures import Point, Size

from algorithms.configuration.entities.entity import Entity
from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.entities.obstacle import Obstacle
from algorithms.configuration.entities.trace import Trace
from algorithms.configuration.maps.map import Map
from algorithms.configuration.maps.bresenhams_algo import bresenhamline

from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest
from algorithms.classic.sample_based.core.graph import gen_cyclic_graph, CyclicGraph

### Reference Classes and Utilities

## Class `Point`:
A point has a variable number of coordinates specified in the constructor. Each point is immutable.

Constructor: Point(*ords), Point(x=..., y=..., z=...)  # keyword-style initialization also supported
e.g. Point(3, 4) or Point(x=3, y=4)

Requires at least 2 dimensions.
Automatically determines whether the point is float or integer based on input values.

Property Attributes:
- x: float | int : First coordinate, horizontal axis in 2D. (e.g. `Point(1, 2).x` returns `1`)
- y: float | int : Second coordinate, vertical axis in 2D. (e.g. `Point(1, 2).y` returns `2`)
- z: float | int (only present in 3D or higher) : Third coordinate. Accessing this on 2D points raises an assertion. (e.g. `Point(1, 2, 3).z` returns `3`)
- n_dim: int : The number of dimensions in the point. (e.g. `Point(1, 2, 3).n_dim` returns `3`)
- values: Tuple[float | int, ...] : The full set of coordinates. (e.g. `Point(1, 2, 3).values` returns `(1, 2, 3)`)
- is_float: bool Indicates whether any of the coordinates are floating-point values. (e.g. `Point(1.0, 2.0).is_float` returns `True`, while `Point(1, 2).is_float` returns `False`)

Methods:
- to_tensor() -> torch.Tensor : Returns a float tensor representation of the point. (e.g. `Point(1, 2).to_tensor()` returns a tensor with values `[1.0, 2.0]`)
- __getitem__(idx) -> float | int : Allows indexing to access coordinates (e.g., `point[0]` for x).

## Class `Size`:
This tuple is used to describe the size of an object in n dimensions. The first three dimensions can be referred to as width, height, and depth, which correspond to x, y, and z accordingly.

Constructor: Size(*ords), Size(width=..., height=..., depth=...)  # keyword-style initialization also supported
e.g. Size(3, 4) or Size(width=3, height=4)

Property attribute:
- values: Tuple[int, ...] (e.g. `Size(3, 4).values` returns `(3, 4)`)
- width: int (e.g. `Size(3, 4).width` returns `3`)
- height: int (e.g. `Size(3, 4).height` returns `4`)
- depth: int (e.g. `Size(3, 4, 5).depth` returns `5`)
- n_dim: int (e.g. `Size(3, 4, 5).n_dim` returns `3`)

Methods:
- to_tensor() -> torch.Tensor : Returns a float tensor representation of the size. (e.g. `Size(1, 2).to_tensor()` returns a tensor with values `[1.0, 2.0]`)
- __getitem__(idx) -> float | int : Allows indexing to access coordinates (e.g., `Size[0]` for width).

## Class `Entity`:
This class is the main class for all objects found on a map. Every object from the map must extend this class.

Constructor: Entity(position: Point, radius: float = 0)

Attributes:
- position: Point  The position of the entity in the grid. (e.g. `entity.position` returns a Point object)
- radius: float - The radius of the entity, used for collision detection and movement constraints. (e.g. `entity.radius` returns a float value)


## Class `Agent`, `Goal`, `Obstacle`, and `Trace`:
Represents the main entity. It is currently the only entity that can be moved and it should be unique to the map. it inherits from `Entity` class.
Constructor: Agent(position: Point, radius: float = 0), Goal(position: Point, radius: float = 0), Obstacle(position: Point, radius: float = 0), Trace(position: Point, radius: float = 0)

Attributes:
- position: Point  The position of the entity in the grid.
- radius: float - The radius of the entity, used for collision detection and movement constraints.

usasge example:
agent = Agent(Point(0, 0), radius=1.0)
goal = Goal(Point(5, 5), radius=1.0)
obstacle = Obstacle(Point(2, 2), radius=0.5)
trace = Trace(Point(3, 3), radius=0.1)

agent.position  # Returns Point(0, 0)
agent.position.value  # Returns (0, 0)
goal.radius     # Returns 1.0
obstacle.radius # Returns 0.5
trace.position  # Returns Point(3, 3)

## Class `Map`: This class represents the environment in which the agent can move with dimensions corresponding to the agent. It contains 1 agent, 1 goal and multiple obstacles. All maps must inherit this class.

Constructor: Map(agent: Agent, goal: Goal, obstacles: List[Obstacle])

Attributes:
- agent: Agent
- goal: Goal
- obstacles: List[Obstacle]
- trace: List[Trace]
- size: Size

Agent, Goal, Obstacle and Trace are classes that represent the agent, goal, obstacle and trace positions in the grid. Access to their attributes positions and radius. and inherit from `Entity` class
(e.g. `Map.agent.position.value` = (0, 0), `Map.agent.radius` = 1.0, `Map.size.values` = (10, 10), `Map.size.width` = 10, `Map.size.height` = 10)

methods:
- get_obstacle_bound(self, obstacle_start_point: Point, visited: Optional[Set[Point]] = None) -> Set[Point]: Method for finding the bound of an obstacle by doing a fill (e.g. `Map.get_obstacle_bound(Point(2, 2), {Point(3, 5), Point(3,3)})` returns {Point(2, 2)}).
- get_move_index(self, direction: List[float]) -> int: Method for getting the move index from a direction, return The index as an integer
- get_move_along_dir(self, direction: List[float]) -> Point: Method for getting the movement direction from a normal direction, return The movement direction as a Point
- get_line_sequence(self, frm: Point, to: Point) -> List[Point]: Bresenham's line algorithm, Given coordinate of two n dimensional points. The task to find all the intermediate points required for drawing line AB.
- is_valid_line_sequence(self, line_sequence: List[Point]) -> bool
- is_goal_reached(self, pos: Point, goal: Goal = None) -> bool: Method that checks if the position coincides with the goal, return If the goal has been reached from the given position
- is_agent_in_goal_radius(self, agent_pos: Point = None, goal: Goal = None) -> bool
- is_agent_valid_pos(self, pos: Point) -> bool: Checks if the point is a valid position for the agent, return If the point is a valid position for the agent
- get_next_positions(self, pos: Point) -> List[Point]: Returns the next available positions, return A list of positions
- get_next_positions_with_move_index(self, pos: Point) -> List[Tuple[Point, int]]: Returns the next available positions with move index, return A list of positions with move index
- get_movement_cost(self, frm: Union[Point, Entity] = None, to: Union[Point, Entity] = None) -> float: Returns the movement cost from one point to another, return The movement cost as a float
- get_movement_cost_from_index(self, idx: int, frm: Optional[Point] = None) -> float: Returns the movement cost from one point to another using the move index
- get_distance(frm: Point, to: Point) -> float: Returns the distance between two points (this is staticmethod function, it can use like this `Map.get_distance(frm, to)` or `self._get_grid().get_distance(frm, to)`)

## `Vertex` class:
The `Vertex` class represents a node in the path planning graph. It stores the position of the vertex, its children and parents, and the cost associated with it. The class provides methods to add child and parent vertices, allowing for the construction of a directed graph structure.

Constructor: Vertex(position: Point, store_connectivity: bool = False)
(e.g. vertex = Vertex(Point(1, 2), store_connectivity=True))

Attributes:
- position: Point = position
- cost: float = None
- children: Set['Vertex'] = set()
- parents: Set['Vertex'] = set()
- connectivity: Set['Vertex'] = store_connectivity
- aux: Dict[Any, Any] = {}

This Attributes cans access the vertex position, cost, children, parents, connectivity and auxiliary data.
(e.g. `.position`, `.cost`, `.children`, `.parents`, `.connectivity`, `.aux`)

Property Attributes:
- cost(self, val:float) - Allows external code to set the `cost` value of the vertex (e.g., `vertex.cost = 1.5`, best_cost = vertext.cost).
- position(self, val:Point) - Allows external code to set the `position` value of the vertex (e.g., `vertex.position = Point(3, 4)`, vertex.position.values = (3, 4)).
- parents(self) - Returns the parents of the vertex as a set (e.g., `vertex.parents` returns a set of parent vertices).
- children(self) - Returns the children of the vertex as a set (e.g., `vertex.children` returns a set of child vertices).

Methods:
- add_child(self, child: 'Vertex') -> None: Adds a child vertex to the current vertex. (e.g. `vertex.add_child(Vertex(Point(2, 3))`)
- add_parent(self, parent: 'Vertex') -> None: Adds a parent vertex to the current vertex. (e.g. `vertex.add_parent(Vertex(Point(0, 1))`)
- set_child(self, child: 'Vertex'): Sets a child vertex for the current vertex. (e.g. `vertex.set_child(Vertex(Point(2, 3))`)
- set_parent(self, parent: 'Vertex'): Sets a parent vertex for the current vertex. (e.g. `vertex.set_parent(Vertex(Point(0, 1))`)
- visit_children(self, f: Callable[['Vertex'], bool]) -> None: Visits all child vertices and applies a function to them. 
- visit_parents(self, f: Callable[['Vertex'], bool]) -> None: Visits all parent vertices and applies a function to them.

## Class `Forest`:
The `Forest` class represents a collection of trees (graphs) in the path planning algorithm. It allows for the management of multiple root vertices and provides methods to add edges, find nearest vertices, and retrieve vertices within a certain radius.

Constructor: Forest(root_vertex_start: Vertex, root_vertex_goal: Vertex, root_vertices: List[Vertex] = None)
e.g. `forest = Forest(Vertex(Point(0, 0)), Vertex(Point(5, 5)))`, `graph = gen_forest(self._services, Vertex(self._get_grid().agent.position), Vertex(self._get_grid().goal.position), [])`

Attributes:
- root_vertex_start: Vertex
- root_vertex_goal: Vertex
- root_vertices: List[Vertex]
- size: int

This Attributes cans access the root vertices, size of the forest and root vertices for start and goal.
(e.g. `graph.root_vertex_start`, `graph.root_vertex_goal`, `graph.root_vertices`, `graph.size`)

Methods:
- reverse_root_vertices(self) -> None: Reverses the order of root vertices in the forest.
- get_random_vertex(self, root_vertices: List[Vertex]) -> Vertex: Returns a random vertex from the specified root vertices.
- get_nearest_vertex(self, root_vertices: List[Vertex], point: Point) -> Vertex: Returns the nearest vertex to a given point from the specified root vertices.
- get_vertices_within_radius(self, root_vertices: List[Vertex], point: Point, radius: float) -> List[Vertex]: Returns a list of vertices within a specified radius from a given point in the specified root vertices.
- add_edge(self, parent: Vertex, child: Vertex): Adds an edge between two vertices in the forest.
- remove_edge(self, parent: Vertex, child: Vertex): Removes an edge between two vertices in the forest.
- walk_dfs_subset_of_vertices(self, root_vertices_subset: List[Vertex], f: Callable[[Vertex], bool]): Performs DFS from each vertex in the subset, applying f to all descendants.
- walk_dfs(self, f: Callable[[Vertex], bool]): Performs DFS from all root vertices, applying f to all descendants.

## Class `CyclicGraph`:
The `CyclicGraph` class is a specialized version of the `Forest` class that allows for the addition of edges between vertices, enabling cyclic connections. It inherits from the `Forest` class and provides additional functionality for managing cyclic relationships between vertices.

Constructor: CyclicGraph(root_vertex_start: Vertex, root_vertex_goal: Vertex, root_vertices: List[Vertex] = None)
e.g. `cyclic_graph = CyclicGraph(Vertex(Point(0, 0)), Vertex(Point(5, 5)))`, `graph = gen_cyclic_graph(self._services, Vertex(self._get_grid().agent.position), Vertex(self._get_grid().goal.position), [])`

Attributes:
- root_vertex_start: Vertex
- root_vertex_goal: Vertex
- root_vertices: List[Vertex]
- size: int

This Attributes cans access the root vertices, size of the forest and root vertices for start and goal.
(e.g. `graph.root_vertex_start`, `graph.root_vertex_goal`, `graph.root_vertices`, `graph.size`)

Methods:
- reverse_root_vertices(self) -> None: Reverses the order of root vertices in the forest.
- get_random_vertex(self, root_vertices: List[Vertex]) -> Vertex: Returns a random vertex from the specified root vertices.
- get_nearest_vertex(self, root_vertices: List[Vertex], point: Point) -> Vertex: Returns the nearest vertex to a given point from the specified root vertices.
- get_vertices_within_radius(self, root_vertices: List[Vertex], point: Point, radius: float) -> List[Vertex]: Returns a list of vertices within a specified radius from a given point in the specified root vertices.
- walk_dfs_subset_of_vertices(self, root_vertices_subset: List[Vertex], f: Callable[[Vertex], bool]): Applies f to each vertex in the subset; stops early if f returns False.
- walk_dfs(self, f: Callable[[Vertex], bool]): Applies f to each root vertex; stops early if f returns False.

`Bresenhamline` method
- bresenhamline(start, end, max_iter=-1): Returns a list of points from (start, end] by ray tracing a line b/w the points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed
'''
        self.combined_sample_based_algorithm = '''
`SampleBasedAlgorithm` class:

Methods:
- _init_displays(self) -> None: Initializes the map displays for the algorithm
- key_frame(self, *args, **kwargs) -> None: Marks a key frame for animations
- _get_grid(self) -> Map: Gets the Map object, See Class `Map` Description
- move_agent(self, to: Point) -> None: Moves the agent to the specified point
'''

    def get_task(self):
        return self.prompt_task
    def get_objective(self):
        return self.objective
    def get_constraints(self):
        return self.constraints
    def get_hier_constraints(self):
        return self.hier_constraints
    def get_package_info(self):
        return self.package_info
    def get_inherit_prompt(self):
        return f'''
The class you generate must inherit from `SampleBasedAlgorithm`.

{self.combined_sample_based_algorithm}
'''