import re
import time
from ...llm.interface_LLM import InterfaceLLM
import ast
import sys
from .function_parser import FunctionParser
import astunparse
import textwrap

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode, prompts, hier_gen, **kwargs):
        self.prompt_task         = prompts.get_task()
        self.prompt_objective    = prompts.get_objective()
        self.prompt_constraints  = prompts.get_constraints() if not hier_gen else prompts.get_hier_constraints()
        self.package_info        = prompts.get_package_info()
        self.inherit_prompt = prompts.planning_code.get_inherit_prompt()

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking
        self.hier_gen = hier_gen

        # set operator instruction
        self.e1 = ''
        self.e2 = ''
        self.m1 = ''
        self.m2 = ''
        self.m3 = ''

        if 'no_lm' not in kwargs.keys():
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def get_prompt(self, indivs, op=None):
        prompt_indiv = ""
        if len(indivs)>1:
            prompt_indiv="I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"
            for i in range(len(indivs)):    
                prompt_indiv=prompt_indiv+f"No.{str(i+1)} algorithm and the corresponding code are: \nAlgorithm description: {indivs[i]['algorithm']}\nCode:\n{indivs[i]['code']}\n"
        else: 
            prompt_indiv=f"Reference Implementation:\nAlgorithm description: {indivs[0]['algorithm']}\nCode:\n{indivs['code']}\n"
            
            prompt_content= self.prompt_task+"\n"+\
                "Below is reference information describing available classes and utility functions to help you understand the purpose and capabilities of the imported components.\n"+\
                self.package_info+"\n"+\
                self.inherit_prompt+"\n"+\
                prompt_indiv+\
                f'Instruction : {getattr(self, op)}' if hasattr(self, op) else ''+\
                self.prompt_objective+"\n"+\
                self.prompt_constraints

        return prompt_content
    
    def get_fgen_prompt(self, f_name, f_assign, code_string):
        prompt = f'You are provided with a main function that references helper functions which have not yet been defined.'\
        +'Below is supplementary reference information describing available classes and utility functions used in the provided code.'+self.package_info + "\n" + self.inherit_prompt + "\n"\
        + f"""
You are provided with a function call `{f_name}` that is used in the following code but has not been defined yet:

---
{code_string}
---

Generate only the full definition of `{f_name}`, based on how it is used in the code. 
If the function you generate calls any additional undefined helper functions, do not generate them now — just return the definition of `{f_name}`.
""" + '\n' + f'Implement function {f_name} called like {f_assign}' + "Do not give additional explanations."
        return prompt

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

        response = self.interface_llm.get_response(prompt_content)

        class_name = "PathPlanning"

        algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        algorithm = algorithm_match.group(1).strip() if algorithm_match else ""
        if algorithm_pass: algorithm = 'None algorithm'

        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        
        code = self._extract_class_code(code, class_name)

        code = re.sub(r'([\'"]{3})\s*\{.*?\}\s*\1', '', code, flags=re.DOTALL)


        # print(">>> !!!!!!!!!!!!!!!!!!!check response : \n", response)
        # print(">>> !!!!!!!!!!!!!!!!!!!check designed algorithm: \n", algorithm)
        # print(">>> !!!!!!!!!!!!!!!!!!!check designed code: \n", code)
        
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            class_name = "PathPlanning"

            algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
            algorithm = algorithm_match.group(1).strip() if algorithm_match else ""

            code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            
            code = self._extract_class_code(code, class_name)

            code = re.sub(r'([\'"]{3})\s*\{.*?\}\s*\1', '', code, flags=re.DOTALL)

            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm
        code = code

        code_all = code

        return [code_all, algorithm]
    
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

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.hier_gen: 
            defined_funcs, generated_funcs = set(), list()
            code_all = self.get_function(code_all, defined_funcs, generated_funcs)

        code_all = f"{code_all}"

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    

    
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
        code_string = '# Code:\n```python\n' + code_string + '\n```'
        error_string = '# Error \n ```text\n' + error_string + '\n```'

        return 'Below is supplementary reference information describing available classes and utility functions used in the provided code.'+self.package_info + "\n" + self.inherit_prompt + "\n" + prompt + code_string + error_string
    
    def trouble_shoot(self, code_string, error_string):
        prompt = self._get_trouble_shoot_prompt(code_string, error_string)
        [code, _] = self._get_alg(prompt, algorithm_pass=True)
        return code

