# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "path_planning":
            from .optimization.classic_benchmark_path_planning import run
            self.prob = run.PATHPLANNING()
            print("- Prob "+paras.problem+" loaded ")

        else:
            print("problem "+paras.problem+" not found!")

    def get_problem(self):
        return self.prob
