from eoh.methods.eoh.classic_planning_method import GetPlanningCode

class GetPrompts():
    def __init__(self):
        self.planning_code = GetPlanningCode()
        
        self.prompt_task = "Your task is to design and implement Path planning algorithm. Main objective is to improve the path planning performance below."

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
- Implement it in Python.
- You DO NOT NEED to declare the any imports.
- When connecting nodes and adding edges in the planner, always perform two critical checks:
1.Collision check for the node position: Ensure that the new node itself does not lie inside any obstacle.
2.Edge-obstacle intersection check: Before adding an edge between two nodes, verify that the straight-line path between them does not intersect or pass through any obstacle.
- After code generation, you must review the code to ensure it is syntactically correct, logically coherent, and executable within the expected environment.
- At the top of your response, write an description of the algorithm in curly braces {}, followed by a concise explanation of the planning mechanism in round parentheses ().
- Both the description and the planning mechanism should be placed outside and above the code block.
- Output the code block containing the implementation only.
⚠️ Do not give additional explanations.
'''

        self.architecture_info = '''
Refer to below architecture:
!!!!!generate Only class Node and Planner!!!!!!
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
    def __init__(self, max_iter: int = 5000):
        self.max_iter = max_iter

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
'''

    def get_task(self):
        return self.prompt_task
    def get_objective(self):
        return self.objective
    def get_constraints(self):
        return self.constraints
    def get_architecture_info(self):
        return self.architecture_info