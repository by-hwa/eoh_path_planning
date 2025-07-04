import re
import time
from ...llm.interface_LLM import InterfaceLLM
import ast
import sys

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode, prompts, hier_gen, **kwargs):
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        self.prompt_role         = prompts.get_role()
        self.prompt_objective    = prompts.get_objective()
        self.prompt_constraints  = prompts.get_constraints() if not hier_gen else prompts.get_hier_constraints()
        self.package_info        = prompts.get_package_info()
        self.helper_function     = prompts.get_helper_function()
        self.inherit_prompt = prompts.planning_code.get_inherit_prompt()

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking
        self.hier_gen = hier_gen

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def get_prompt_path_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
self.package_info + "\n" + self.inherit_prompt + "\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+"\n"\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
+ self.prompt_objective + "\n" + self.prompt_constraints + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_path_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
self.package_info + "\n" + self.inherit_prompt + "\n" \
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+"\n"\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Identify the common backbone idea in the provided algorithms." \
+ self.prompt_objective + "\n" + self.prompt_constraints + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_time(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n"\
+self.package_info + "\n" + self.inherit_prompt + "\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved time by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.prompt_objective + "\n" + self.prompt_constraints + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_distance(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n"\
+self.package_info + "\n" + self.inherit_prompt + "\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved distance by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_objective + "\n" + self.prompt_constraints + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_smoothness(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n"\
+self.package_info + "\n" + self.inherit_prompt + "\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved smoothness by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_objective + "\n" + self.prompt_constraints  + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_clearance(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n"\
+self.package_info + "\n" + self.inherit_prompt + "\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved clearance by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_objective + "\n" + self.prompt_constraints  + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_memory(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"Below is supplementary reference information describing available classes and utility functions used in the provided code.\n" \
+ "This context is intended to help you understand the purpose and capabilities of the imported components. \n" +\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n"\
+self.package_info + "\n" + self.inherit_prompt + "\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved computing memory by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_objective + "\n" + self.prompt_constraints + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_fgen_prompt(self, code_string):
        prompt = ''
        # TODO
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
    
    def get_function(self, prompt_content):


        response = self.interface_llm.get_response(prompt_content)
        code_string = ''
        #TODO Extract function

        return code_string
    
    def debug_info(self, function_name, prompt_content, algorithm, code_all):
        print(f"\n >>> check prompt for creating algorithm using [ {function_name} ] \n", prompt_content )
        print("\n >>> check designed algorithm: \n", algorithm)
        print("\n >>> check designed code: \n", code_all)

    def e1(self,parents):
        prompt_content = self.get_prompt_path_e1(parents)

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.hier_gen: 
            pass # TODO
            helper_function = self.get_function(code_all)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    
    def e2(self,parents):
        prompt_content = self.get_prompt_path_e2(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    
    def m_time(self, parents):
        prompt_content = self.get_prompt_time(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    
    def m_distance(self, parents):
        prompt_content = self.get_prompt_distance(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    
    def m_smoothness(self, parents):
        prompt_content = self.get_prompt_smoothness(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)
        return [code_all, algorithm]
    
    def m_clearance(self, parents):
        prompt_content = self.get_prompt_clearance(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode: self.debug_info(sys._getframe().f_code.co_name, prompt_content, algorithm, code_all)

        return [code_all, algorithm]
    
    def m_memory(self, parents):
        prompt_content = self.get_prompt_memory(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)

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

        constraint = '''
# Constraint
- Maintain original logic and style as much as possible.
- Use existing helper functions if available.
- Do not introduce unnecessary external libraries.
'''

        return prompt + code_string + error_string
    
    def trouble_shoot(self, code_string, error_string):
        prompt = self._get_trouble_shoot_prompt(code_string, error_string)
        [code, _] = self._get_alg(prompt, algorithm_pass=True)
        return code

