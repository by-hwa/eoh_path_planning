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

        self.helper_funtion = '''
from algorithms.classic.sample_based.core.vertex import Vertex
from algorithms.classic.sample_based.core.graph import gen_forest, Forest

    start_vertex = Vertex(self._get_grid().agent.position)
    start_vertex.cost = 0
    goal_vertex = Vertex(self._get_grid().goal.position)

    self._graph = gen_forest(self._services, start_vertex, goal_vertex, [])

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

'''