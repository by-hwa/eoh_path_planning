import re
import time
from unittest import loader
from ...llm.interface_LLM import InterfaceLLM
import ast
import sys
from .function_parser import FunctionParser
import astunparse
import textwrap
import os, json, random

# For query transformation
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# For basic RAG implementation
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Evolution():
    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode, prompts, database_mode=True, interactive_mode=True,**kwargs):
        self.prompt_task         = prompts.get_task()
        self.prompt_objective    = prompts.get_objective()
        self.prompt_constraints  = prompts.get_constraints()
        self.architecture_info   = prompts.get_architecture_info()
        self.generation_role   = prompts.get_generation_role()

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking
        self.database_mode = database_mode
        self.interactive_mode = interactive_mode
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.time_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "time_analysis_db.json")
        self.path_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "path_analysis_db.json")
        self.smoothness_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "smoothness_analysis_db.json")
        
        self.entire_pop_path = os.path.join(self.base_dir, "results", "pops", "evaluated_entire_population_generation.json")
        self.init_pop_path = os.path.join(self.base_dir, "results", "pops", "population_generation_0.json")

        # set operator instruction
        self.e1 = '''Design a brand new algorithm from scratch.'''
        self.e2 = '''Design a new algorithm that consolidates the parents effective strategies.'''
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

        self.embedder = OpenAIEmbeddings(api_key=self.api_key,
                              model="text-embedding-3-small")
        
        self.db_dict = {'planning time': self.load_database(self.time_analysis_db_path),
                    'path length': self.load_database(self.path_analysis_db_path),
                    'path smoothness': self.load_database(self.smoothness_analysis_db_path),
                    }
        
        self.conversation_log = []
        self.critic_conversation_log = []
        
        self.time, self.path, self.smooth = 0, 0, 0
        
    def reset_log(self):
        self.conversation_log = []
        
    def reset_critic_log(self):
        self.critic_conversation_log = []

    def logging(self, name, content):
        message = {
            "role": "assistant",
            "name": name,
            "content": content
        }
        self.conversation_log.append(message)
        
    def critic_logging(self, name, content):
        message = {
            "role": "assistant",
            "name": name,
            "content": content
        }
        self.critic_conversation_log.append(message)

    def transform_msg(self, role, name, content):
        message = {
            "role": role,
            "name": name,
            "content": content
        }
        return message
    
    def _get_improv_from_json(self, path):
        ti, li, si = [], [], []
        with open(path, "r") as f:
            datas = json.load(f)
            if not len(datas):return [[],[],[]]

        for data in datas:
            ti.append(data["time_improvement"])
            li.append(data["length_improvement"])
            si.append(data["smoothness_improvement"])

        return [ti, li, si]
    
    def init_max_impv(self):
        t,l,s = self._get_improv_from_json(self.init_pop_path)
        
        self.time = max(t)
        self.path = max(l)
        self.smooth = max(s)
    
    def update_max_impv(self):
        
        t,l,s = self._get_improv_from_json(self.entire_pop_path)
        
        self.time = max(self.time, max(t)) if t else self.time
        self.path = max(self.time, max(l)) if l else self.path
        self.smooth = max(self.time, max(s)) if s else self.smooth
        

    def critic_agent(self, indiv):
        message = []
        prompt = f'''
below is parents code:
{indiv['code']}

Peak performance of the population:
    time_improvement: avg - {self.time},
    length_improvement: avg - {self.path},
    smoothness_improvement: avg - {self.smooth}

Performance :
    time_improvement: {[round(m['time_improvement'],2) for m in indiv['other_inf']]}  avg - {indiv['time_improvement']},
    length_improvement: {[round(m['length_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['length_improvement']},
    smoothness_improvement: {[round(m['smoothness_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['smoothness_improvement']}
    
'''+'''
## you answer below template
[problem of path time aspect]
<problem of path length aspect>
{problem of smoothness aspect}

-Replace the text inside [], <>, {} with the actual problems found in the code.
-Each placeholder must contain only the problems related to that specific metric.
-Please keep the content brief
'''
        message.append(self.transform_msg("system", "system", "You are a senior reviewer for sampling-based path planning (RRT/RRT*/RRT*-Connect/PRM variants). Your job is to diagnose why the given parents code harms (1) planning time, (2) path length, (3) smoothness."))
        message.append(self.transform_msg("user", "critic_agent", prompt))

        response = self.interface_llm.get_response(message)
        time_match = re.search(r"\[(.*?)\]", response, re.DOTALL)
        length_match = re.search(r"\<(.*?)\>", response, re.DOTALL)
        smooth_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        print(response)
        critic = {
            'planning time': time_match.group(1).strip() if time_match else "",
            'path length': length_match.group(1).strip() if length_match else "",
            'path smoothness': smooth_match.group(1).strip() if smooth_match else ""
        }
        contents = f'''
        Peak performance of the population:
            time_improvement: avg - {self.time},
            length_improvement: avg - {self.path},
            smoothness_improvement: avg - {self.smooth}
        
        problem from parents code:
        - Planning time: {critic['planning time']}
        - Path length: {critic['path length']}
        - Path smoothness: {critic['path smoothness']}
        
        and it's Performance :
            time_improvement: {[round(m['time_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['time_improvement']},
            length_improvement: {[round(m['length_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['length_improvement']},
            smoothness_improvement: {[round(m['smoothness_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['smoothness_improvement']}
        '''
        if self.interactive_mode:
            self.logging("critic_agent", contents)
        
        return critic

    def load_database_dict(self):
        self.db_dict = {'planning time': self.load_database(self.time_analysis_db_path),
                    'path length': self.load_database(self.path_analysis_db_path),
                    'path smoothness': self.load_database(self.smoothness_analysis_db_path),
                    }

    def load_database(self, file_path):
        documents = self.load_documents(file_path)
        if documents:
            vector_db = FAISS.from_documents(documents, self.embedder)
            return vector_db
        return None

    def load_documents(self, file_path):
        def metadata_func(record: dict, metadata: dict) -> dict:
            return {"solution": record.get("solution", "")}
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".[].analysis",  # This extracts the content field from each array item
                content_key="problem",
                text_content=False,
                metadata_func=metadata_func
            )
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading documents from {file_path}: Empty database.")
            print("Exception:", str(e))
            return None

    def load_analysis(self):
        for k, (path, analysis) in self.analysis_db_dict.items():
            with open(path, "r") as f:
                data = json.load(f)
                self.analysis_db_dict[k] = (path, data)
    # analysis agent
    def get_analysis(self, improvment_metric, parents, offspring):
        message = []
        parent_block_lines = []
        for i, p in enumerate(parents, start=1):
            parent_block_lines.append(f"- Parents #{i} algorithm:\n```python\n{p['code']}\n``` \nPerformance :\n    time_improvement: {[round(m['time_improvement'],2) for m in p['other_inf']]},\n    length_improvement: {[round(m['length_improvement'],2) for m in p['other_inf']]},\n    smoothness_improvement: {[round(m['smoothness_improvement'],2) for m in p['other_inf']]}\n")
        parent_block = "\n\n".join(parent_block_lines)

        prompt = f"""
The following are the structural differences between multiple or single parent path planning algorithms and one offspring algorithm:

- Parents algorithms:
{parent_block}

- Offspring algorithm:
```python
{offspring['code']}
```
Peak performance of the population:
    time_improvement: avg - {self.time},
    length_improvement: avg - {self.path},
    smoothness_improvement: avg - {self.smooth}

Offspring Performance :
    time_improvement:avg - {offspring['time_improvement']},
    length_improvement: avg - {offspring['length_improvement']},
    smoothness_improvement: avg - {offspring['smoothness_improvement']}

Improved performance metric: {improvment_metric}

[Instructions]
- DO NOT GIVE ANY EXPLANATION JUST BELOW INFORMATION
- DO NOT mention specific parent identifiers, numbers, or names (e.g., “parent#2” or algorithm names).
- Describe problems and improvements only in **general terms** (e.g., “inefficient exploration in cluttered regions”, “lack of edge rewiring”).
- Aggregate observations if needed, but keep them generic (“some variants”, “several approaches”).

Please analyze and output the results in the following format:

<Problem of the parents algorithm:
   - ...
   - ...
 >  
 
[1. Primary contributors to the performance improvement:
   - ...
2. Expected mechanism of impact:
   - ...]
"""
        message.append(self.transform_msg("system", "system", "You are the Analysis Agent. Your job is to compare parent vs offspring path planning algorithms and explain why performance improved. Always output in the required bullet-point format, nothing else."))
        message.append(self.transform_msg("user", "analysis_agent", prompt))
        print("Waiting analysis response")
        analysis = self.interface_llm.get_response(self.conversation_log+message)
        
        prob_match = re.search(r"\<(.*?)\>", analysis, re.DOTALL)
        sol_match = re.search(r"\[(.*?)\]", analysis, re.DOTALL)
        
        prob = prob_match.group(1).strip() if prob_match else "",
        sol = sol_match.group(1).strip() if sol_match else ""
        
        if prob and sol:
            if self.interactive_mode:
                self.logging("analysis_agent", f"Problem: {prob} \n Solution: {sol}")
            return {'problem': prob, 'solution': sol}
        
        else: return None
    
    def get_discussion(self):        
        instruction = '''
[Input]
- Critic chat logs for each metric ("planning time", "path length", "path smoothness"). 
  Each critic log contains only <problems> about the parent algorithm, without improvement suggestions.
- Performance summary:
  * Parent algorithm performance for each metric
  * Best performance achieved so far by any generated algorithm

[Task]
1. Read the critic chat logs and performance summary carefully.
2. Simulate an internal debate among the critics to propose possible improvements for the identified problems.
3. Decide which **metric** requires improvement the most, considering both:
   - Critics’ reported problems
   - Gaps between parent performance and the best achieved performance
4. From the debate outcome, extract:
    - The most critical [performance metric] to improve (must be wrapped in square brackets []).
   - The main <problem> for this metric (must be wrapped in angle brackets <>).
   - The most appropriate {improvement method} generated during the debate (must be wrapped in curly braces {}).
5. Ignore minor or repetitive issues. Focus on the strongest consensus.

[Reference]
Peak performance of the population:
    time_improvement: avg - {self.time},
    length_improvement: avg - {self.path},
    smoothness_improvement: avg - {self.smooth}

[Output format]
[planning time|path length|path smoothness]
<problem (1–5 sentences)>
{improvement method (1–5 sentences)}

[Constraints]
- Output must have exactly 3-5 lines, no blank lines.
- The improvement must be generated by you, based on debate reasoning and performance comparison.
- Do not add explanations, notes, or extra text outside the required format.
'''

        message1 = []
        message1.append(self.transform_msg("system", "system", "You are a debate summarizer agent. You must read critic logs and performance summaries, simulate a discussion, and output exactly 3 lines in the required format."))

        message2 = []
        message2.append(self.transform_msg("user", "discussion_agent", instruction))
        
        response = self.interface_llm.get_response(message1+self.critic_conversation_log+message2)
        print("Discussion response: ", response)

        metric_match = re.search(r"\[(.*?)\]", response, re.DOTALL)
        prob_match = re.search(r"\<(.*?)\>", response, re.DOTALL)
        sol_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        
        metric = metric_match.group(1).strip() if metric_match else ""
        prob = prob_match.group(1).strip() if prob_match else ""
        sol = sol_match.group(1).strip() if sol_match else ""

        return metric, prob, sol

    def get_prompt(self, indivs, op=None):
        op='None' # TODO for test
        
# Peak performance of the population:
#     time_improvement: avg - {self.time},
#     length_improvement: avg - {self.path},
#     smoothness_improvement: avg - {self.smooth}
        prompt_indiv = ""
        if len(indivs)>1:
            prompt_indiv="I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"
            for i in range(len(indivs)):    
                prompt_indiv=prompt_indiv+f'''No.{str(i+1)} algorithm and the corresponding code are: 
                Algorithm description: {indivs[i]['algorithm_description']}
                Planning Mechanism:\n{indivs[i]['planning_mechanism']}
                Code:\n{indivs[i]['code']}
                
'''
        else: 
            prompt_indiv=f"Reference Implementation:\nAlgorithm description: {indivs[0]['algorithm_description']}\nPlanning Mechanism:\n{indivs[0]['planning_mechanism']}\nCode:\n{indivs[0]['code']}\n"
        
        prompt_indiv =f'''
Reference
Peak performance of the population:
    time_improvement: avg - {self.time},
    length_improvement: avg - {self.path},
    smoothness_improvement: avg - {self.smooth}
'''
        analysis_info=''

        self.reset_critic_log()
        
        for i, indiv in enumerate(indivs):
            gap = self.critic_agent(indiv)
            for k, v in gap.items():
                if v and len(v):
                    perform = f'''
Performance :
    time_improvement: {[round(m['time_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['time_improvement']},
    length_improvement: {[round(m['length_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['length_improvement']},
    smoothness_improvement: {[round(m['smoothness_improvement'],2) for m in indiv['other_inf']]} avg - {indiv['smoothness_improvement']}
                '''
                    self.critic_logging(f"{k.replace(' ', '_')}_critic_agent_{i+1}", v+perform)

        # gap = self.critic_agent(indivs[0]) # TODO: multiple parents
        
        if self.database_mode and len(indivs)>=1:
            metric, prob, sol = self.get_discussion()
            if isinstance(self.db_dict[metric], FAISS):
                results = self.db_dict[metric].similarity_search_with_score(prob, k=3)
                if len(results):
                    n = random.randint(0, len(results)-1)
                    analysis_info += f"The problem of the parents algorithm in terms of {metric} is: {prob}.\n"
                    if (1/(1 + results[n][1])) > 0.5: # TODO: threshold tuning
                        analysis_info += f"The following is a relevant analysis from our database that may help improve {metric}:\n"+\
                                         results[n][0].page_content+results[n][0].metadata.get("solution", "")+"\n"
                    else:
                        if sol:
                            analysis_info += f"The following is a guide for improving {metric}:\n"+sol+"\n"
                    
                    analysis_info += "above knowledge is just for your reference."
                        
        #     for k, (p, a) in self.analysis_db_dict.items():
        #         if gap and gap[k]:
        #             if isinstance(self.db_dict[k], FAISS):
        #                 results = self.db_dict[k].similarity_search(gap[k], k=3)
        #                 print(gap[k])
        #                 if len(results):
        #                     n = random.randint(0, len(results)-1)
        #                     analysis_info += f"\nThe following prompt is intended to analyze how structural differences between two path planning algorithms (parents alg → offspring alg) have contributed to the improvement of a specific performance metric: {k}.\n"+\
        #                     results[n].page_content+"\n"
        #         elif len(a):
        #             n = random.randint(0, len(a)-1)
        #             analysis_info += f"\nThe following prompt is intended to analyze how structural differences between two path planning algorithms (parents alg → offspring alg) have contributed to the improvement of a specific performance metric: {k}.\n"+\
        #             a[n]['analysis']+"\n"

        prompt_content= ''+\
            prompt_indiv+\
            analysis_info+(f'Instruction : {getattr(self, op)}\n' if hasattr(self, op) else 'Generate algorithm')
            # self.architecture_info
            # self.prompt_objective+
            # self.prompt_constraints
        
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
        message = []

        message.append(self.transform_msg("system", "system", self.generation_role))
        message.append(self.transform_msg("user", "generation_agent", prompt_content))

        print("Waiting get_alg response")
        response = self.interface_llm.get_response(self.conversation_log+message)
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

        contents = f'''
        designed algorithm: {algorithm}
        planning mechanism: {mechanism}
        '''

        if self.interactive_mode:
            self.logging("generation_agent", contents)
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
        prompt_content = self.get_prompt(parents, op) # critic agent

        [code_all, mechanism, algorithm] = self._get_alg(prompt_content) # generation agent

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

