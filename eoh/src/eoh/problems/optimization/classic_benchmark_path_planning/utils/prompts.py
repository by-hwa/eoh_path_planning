# from eoh.methods.eoh.classic_planning_method import GetPlanningCode

class GetPrompts():
    def __init__(self):
        # self.planning_code = GetPlanningCode()
        
        self.prompt_task = "Your task is to design and implement Path planning algorithm. Main objective is to improve the path planning performance below."

        self.objective = '''
### Objective:
- ⚠️The system shall produce a path within 1.5 seconds. The generated path must be as short as possible, with both search time and path length minimized, and the search success rate shall be guaranteed.⚠️
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
- Implement it in Python.
- You DO NOT NEED to declare the any imports.
- When connecting nodes and adding edges in the planner, always perform two critical checks:
1.Collision check for the node position: Ensure that the new node itself does not lie inside any obstacle.
2.Edge-obstacle intersection check: Before adding an edge between two nodes, verify that the straight-line path between them does not intersect or pass through any obstacle.
- DO NOT OVER MAP BOUND
- After code generation, you must review the code to ensure it is syntactically correct, logically coherent, and executable within the expected environment.
- At the top of your response, write an description of the algorithm in curly braces {}, followed by a concise explanation of the planning mechanism in angle brackets <>.
- Both the description and the planning mechanism should be placed outside and above the code block.
- Output the code block containing the implementation only.
⚠️ Do not give additional explanations.
'''

        self.architecture_info = '''
Refer to below architecture:
```python
# --- Node class ---
class Node:
    def __init__(self, position, parent=None, cost=0.0):
        self.position = position        # Tuple[float, ...] → 2D: (x,y), 3D: (x,y,z)
        self.parent = parent            # Node or None
        self.cost = cost                # Path cost
        self.children = []
        self.valid = True               # For collision checking etc.

    #### Create additional methods if needed ####

# --- PlannerResult structure ---
class PlannerResult(NamedTuple):
    success: bool                       # Path navigation success or not
    path: List[Tuple[float, ...]]       # Final path from start to goal
    nodes: List[Node]                   # All explored nodes
    edges: List[Tuple[Node, Node]]      # Parent-child connections

# --- Main Planner ---
class Planner:
    def __init__(self, max_iter: int = 5000, step_size: float=5.0):
        self.max_iter = max_iter # Do not Edit
        self.step_size = step_size # if you need, you can change the step size for performance
        #### Place holder: add additional attributes if needed ####
        
    def plan(self, map: Map) -> PlannerResult:
        bounds = map.size                  # Tuple[int, ...]: (W,H) or (W,H,D)
        start_position = map.start         # Tuple[float, ...] (W,H) or (W,H,D)
        goal_position = map.goal           # Tuple[float, ...] (W,H) or (W,H,D)
        obstacles = map.obstacles          # Rectangular blocks: 2D=(x,y,w,h), 3D=(x,y,z,w,h,d)

        is_3d = len(bounds) == 3

        # Core data
        success_state = False # Path navigation success or not
        extracted_path: List[Tuple[float, ...]] = [] # Final path from start to goal
        nodes: List[Node] = [] # All explored nodes
        edges: List[Tuple[Node, Node]] = [] # Parent-child connections

        #### Place holder: path planning logic ####

        return PlannerResult(
            success=success_state,
            path=extracted_path,
            nodes=nodes,
            edges=edges
        )

    def _is_in_obstacle(self, pos, obstacles, is_3d):
        for obs in obstacles:
            if is_3d:
                x, y, z, w, h, d = obs
                px, py, pz = pos
                if x <= px <= x + w and y <= py <= y + h and z <= pz <= z + d:
                    return True
            else:
                x, y, w, h = obs
                px, py = pos
                if x <= px <= x + w and y <= py <= y + h:
                    return True
        return False

    def _is_edge_in_obstacle(self, from_pos, to_pos, obstacles, is_3d, resolution=1.0):
        distance = math.dist(from_pos, to_pos)
        steps = max(1, int(distance / resolution))
        for i in range(steps + 1):
            interp = tuple(from_pos[d] + (to_pos[d] - from_pos[d]) * (i / steps) for d in range(len(from_pos)))
            if self._is_in_obstacle(interp, obstacles, is_3d):
                return True
        return False
```
'''

        self.generation_role = '''
You are the Generation Agent for sampling-based path planning. 
Design a **brand-new** or **similar but improved** algorithm (do not copy parents) that improves planning efficiency, path quality, robustness, success rate, smoothness, and path length, while reducing search time.
You MUST obey the following output contract and constraints:

[Output contract]
1) At the top of your response, write an description of the algorithm in curly braces {}, followed by a concise explanation of the planning mechanism in angle brackets <>.
DO NOT add any other text.

2) Then output ONE Python code block that defines ONLY:
   - class Node
   - class Planner
   (Do not output any other classes, functions, or text outside the block.)
3) Do NOT add any extra explanations, comments outside the required two headers, or trailing text.

[Implementation constraints]
- Language: Python. Do NOT declare any imports.
- Implement a complete, executable Planner.plan(map) that returns PlannerResult as described by the user.
- Always perform BOTH checks before adding any node/edge(This Include in Code template):
  (1) Node collision: new node must not lie inside any obstacle.
  (2) Edge collision: straight-line segment between nodes must not intersect any obstacle.
- Never sample or step outside map bounds (use map.size). Respect 2D/3D rectangular obstacles as provided.
- When connecting nodes and recording edges, keep data coherent (parents/children, costs, edges list) and avoid duplicates.
- Prefer efficient data flow (early exits, incremental best path updates) and mechanisms that plausibly improve the stated objectives.
- After generation, self-verify syntax and logic coherence; ensure the code is runnable in the expected environment.
'''

    def get_task(self):
        return self.prompt_task
    def get_objective(self):
        return self.objective
    def get_constraints(self):
        return self.constraints
    def get_architecture_info(self):
        return self.architecture_info
    def get_generation_role(self):
        return self.generation_role