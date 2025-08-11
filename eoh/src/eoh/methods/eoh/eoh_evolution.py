import re
import time
from ...llm.interface_LLM import InterfaceLLM
import ast
import sys
from .function_parser import FunctionParser
import astunparse
import textwrap
import os, json, random
class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode, prompts, **kwargs):
        self.prompt_task         = prompts.get_task()
        self.prompt_objective    = prompts.get_objective()
        self.prompt_constraints  = prompts.get_constraints()
        self.architecture_info   = prompts.get_architecture_info()

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.time_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "time_analysis_db.json")
        self.path_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "path_analysis_db.json")
        self.smoothness_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "smoothness_analysis_db.json")

        # set operator instruction
        self.e1 = '''Design a brand new algorithm from scratch.'''
        self.e2 = '''Create a hybrid algorithm inspired by multiple existing ones.'''
        self.m1 = '''Modify the structure of an existing algorithm.'''
        self.m2 = '''Tune and reconfigure the parameters of a given algorithm.'''
        self.m3 = '''Simplify to enhance generalization for given algorithm.'''
        self.cross_over = 'Improve the algorithm by minimizing the path length and reducing planning time, while using insights from previously successful heuristics.'
        self.time_expert = 'Improve the algorithm by minimizing reducing planning time, while using insights from previously successful heuristic.'
        self.path_expert = 'Improve the algorithm by minimizing the path length, while using insights from previously successful heuristic.'

        self.time_analysis = []
        self.path_analysis = []
        self.smoothness_analysis = []

        self.analysis_db_dict = {'planning time': (self.time_analysis_db_path, self.time_analysis),
                            'path length': (self.path_analysis_db_path, self.path_analysis),
                            'path smoothness': (self.smoothness_analysis_db_path, self.smoothness_analysis),
                            }
                
        if 'no_lm' not in kwargs.keys():
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def load_analysis(self):
        for k, (path, analysis) in self.analysis_db_dict.items():
            with open(path, "r") as f:
                data = json.load(f)
                self.analysis_db_dict[k] = (path, data)

    def get_analysis(self, improvment_metric, parents, offspring_code):
        parent_block_lines = []
        for i, code in enumerate(parents, start=1):
            parent_block_lines.append(f"- Parent #{i} algorithm:\n```python\n{code}\n```")
        parent_block = "\n\n".join(parent_block_lines)

        prompt = f"""
The following are the structural differences between multiple or single parent path planning algorithms and one offspring algorithm:

- Parents algorithm:
{parent_block}

- Offspring algorithm:
```python
{offspring_code}
```

Improved performance metric: {improvment_metric}

DO NOT GIVE ANY EXPLANATION JUST BELOW INFORMATION
Please analyze and output the results in the following format:

1. Summary of key changes:
   - ...
   - ...
2. Primary contributors to the performance improvement:
   - ...
3. Expected mechanism of impact:
   - ...
"""
        print("Waiting response")
        analysis = self.interface_llm.get_response(prompt)
        return analysis

    def get_prompt(self, indivs, op=None):
        prompt_indiv = ""
        if len(indivs)>1:
            prompt_indiv="I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"
            for i in range(len(indivs)):    
                prompt_indiv=prompt_indiv+f"No.{str(i+1)} algorithm and the corresponding code are: \nAlgorithm description: {indivs[i]['algorithm_description']}\nPlanning Mechanism:\n{indivs[i]['planning_mechanism']}\nCode:\n{indivs[i]['code']}\n"
        else: 
            prompt_indiv=f"Reference Implementation:\nAlgorithm description: {indivs[0]['algorithm_description']}\nPlanning Mechanism:\n{indivs[0]['planning_mechanism']}\nCode:\n{indivs[0]['code']}\n"
        
        analysis_info = ''
        for k, (p, a) in self.analysis_db_dict.items():
            if len(a):
                n = random.randint(0, len(a))
                analysis_info += f"\nThe following prompt is intended to analyze how structural differences between two path planning algorithms (parents alg → offspring alg) have contributed to the improvement of a specific performance metric: {k}.\n"+\
                a[n]['analysis']

        prompt_content= self.prompt_task+"\n"+\
            prompt_indiv+\
            analysis_info+\
            (f'Instruction : {getattr(self, op)}\n' if hasattr(self, op) else 'Generate algorithm')+\
            self.architecture_info+\
            self.prompt_objective+\
            self.prompt_constraints
        
        return prompt_content
    
    def _extract_alg(self, response: str, algorithm_pass=False) -> str:

        algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        algorithm = algorithm_match.group(1).strip() if algorithm_match else ""

        mechanism_match = re.search(r"\<(.*?)\>", response, re.DOTALL)
        mechanism = mechanism_match.group(1).strip() if mechanism_match else ""
        
        if algorithm_pass:
            algorithm = 'None algorithm'
            mechanism = 'None mechanism'

        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        if code_match:
            all_code = code_match.group(1).strip()

        node_code = self._extract_class_code(all_code, "Node")
        node_code = re.sub(r'([\'"]{3})\s*\{.*?\}\s*\1', '', node_code, flags=re.DOTALL)

        code = self._extract_class_code(all_code, "Planner")
        code = re.sub(r'([\'"]{3})\s*\{.*?\}\s*\1', '', code, flags=re.DOTALL)
        code_all = node_code + "\n" + code
   
        return algorithm, mechanism, code_all

    def _extract_class_code(self, code: str, class_name: str) -> str:
        module = ast.parse(code)
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                start_line = node.lineno - 1  # ast uses 1-based indexing
                end_line = node.end_lineno    # requires Python 3.8+
                lines = code.splitlines()
                return '\n'.join(lines[start_line:end_line])
        return ""

    def _get_alg(self,prompt_content, algorithm_pass=False):

        print("Waiting response")
        response = self.interface_llm.get_response(prompt_content)
        algorithm, mechanism, code_all = self._extract_alg(response, algorithm_pass)

        n_retry = 1
        while (len(algorithm) == 0 or len(code_all) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            algorithm, mechanism, code_all = self._extract_alg(response, algorithm_pass)

            if n_retry > 3:
                break
            n_retry +=1

        return [code_all, mechanism, algorithm]
    
    def strip_outer_code_block(self, text: str) -> str:
    # 앞쪽 ```python 또는 ``` 제거
        text = re.sub(r'^\s*```(?:python)?\s*\n?', '', text)
        # 뒤쪽 ``` 제거
        text = re.sub(r'\n?\s*```$', '', text)
        return text

    def _get_func(self, prompt_content):
        prompt_content = self.strip_outer_code_block(prompt_content)
        tree = ast.parse(prompt_content)
        function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return textwrap.indent(''.join([astunparse.unparse(func) for func in function_defs]), "    ")
    
    def get_function(self, code_string, defined_funcs, generated_funcs):

        while True:
            fs, f_assigns = {}, {}
            class_parser = FunctionParser(fs, f_assigns)
            class_parser.visit(ast.parse(code_string))
            
            new_undefined_funcs = [(f_name, fs[f_name]) for f_name in fs if f_name not in defined_funcs]

            if not new_undefined_funcs:
                break

            f_name, f_assign = new_undefined_funcs[0]
            prompt = self.get_fgen_prompt(f_name, f_assign, code_string)
            generated_code = self.interface_llm.get_response(prompt)
            generated_code = self._get_func(generated_code)

            defined_funcs.add(f_name)
            generated_funcs.append(generated_code)
            code_string += f"\n\n{generated_code}"
 
        return code_string
    
    def debug_info(self, function_name, prompt_content, algorithm, code_all):
        print(f"\n >>> check prompt for creating algorithm using [ {function_name} ] \n", prompt_content )
        print("\n >>> check designed algorithm: \n", algorithm)
        print("\n >>> check designed code: \n", code_all)

    def evol(self,parents, op):
        prompt_content = self.get_prompt(parents, op)

        [code_all, mechanism, algorithm] = self._get_alg(prompt_content)

        code_all = f"{code_all}"

        if self.debug_mode: self.debug_info(op, prompt_content, algorithm, code_all) # sys._getframe().f_code.co_name

        return [code_all, mechanism, algorithm]
    

    
    def _get_trouble_shoot_prompt(self, code_string, error_string):
        prompt = '''
You are given a code snippet and its corresponding error output from execution.  
Your task is to:

1. Analyze the cause of the error based on the provided error message.
2. Modify the original code to fix the error.
3. Return the corrected and executable version of the code.

Do not explain the error unless asked.  
Your output should only include the complete fixed code block with the issue resolved.
'''
        import_string = '''
from typing import Tuple, Literal, Union, Optional, List, Dict, NamedTuple, Callable, Any, Set, TYPE_CHECKING, Type
import time
from queue import Queue
import numpy as np
import random
import math
import sys
import os

from eoh.problems.optimization.classic_benchmark_path_planning.utils.architecture_utils import PlannerResult, Map
'''
        code_string = '# Code:\n```python\n' + code_string + '\n```'
        error_string = '# Error \n ```text\n' + error_string + '\n```'

        return 'Below is supplementary reference information describing available classes and utility functions used in the provided code.' + "\n" + prompt + import_string +code_string + error_string + self.prompt_constraints

    def trouble_shoot(self, code_string, error_string):
        prompt = self._get_trouble_shoot_prompt(code_string, error_string)
        [code, _, _] = self._get_alg(prompt, algorithm_pass=True)
        return code

