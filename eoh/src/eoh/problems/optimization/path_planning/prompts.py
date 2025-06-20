class GetPrompts():
    def __init__(self):
        
        self.role = "You are given a reference implementations for path planning algorithms on a discrete grid environment."
        

        # self.prompt_task = "Your task is to improve upon this baseline algorithm and implement an enhanced path planning class named `PathPlanning`."
        self.prompt_task = "Your task is to design and implement an **improved path planning algorithm**, written as a Python class named `PathPlanning`, that is inspired by but not limited to the provided examples."

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
- The description must be inside a brace.
- Implement it in Python.
- Your class must be named `PathPlanning`.
- It must inherit from `SampleBasedAlgorithm`.
- It should work with existing components: `Forest`, `Point`, `Vertex`, etc.
- Include core methods like `_extend`, `_connect`, `_extract_path`, `_find_path_internal`.
### Reusable helper functions (optional to modify or extend):
- `_get_random_sample()`
- `_get_nearest_vertex(...)`
- `_get_new_vertex(...)`
- `_get_grid()`

### You may freely define new helper functions if necessary
- If your approach benefits from additional utility methods (e.g., cost estimation, region sampling, custom distance functions), feel free to create and use them.
### The `_find_path_internal` function is the main function executed for path planning.
'''

        self.prompt_func_name = ""
        self.prompt_func_inputs = ""
        self.prompt_func_outputs = ""


        self.prompt_inout_inf = "You may assume access to the new implementation via the baseline class and its methods, such as `_get_random_sample`, `_get_nearest_vertex`, etc."
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
