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