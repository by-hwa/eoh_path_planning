import re
import time
from ...llm.interface_LLM import InterfaceLLM
class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        self.prompt_role         = prompts.get_role()
        self.prompt_objective    = prompts.get_objective()
        self.prompt_constraints   = prompts.get_constraints()
        self.inherit_prompt = prompts.planning_code.get_inherit_prompt()
        try:
            self.prompt_reference_tree = prompts.get_tree()
            self.prompt_get_code_template = prompts.get_code_template()
        except:
            self.prompt_reference_tree = None
            self.prompt_get_code_template = None


        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def get_prompt_i1(self):
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"\
+prompt_code_template\
+reference_sentence\
+"Do not give additional explanations."
    # +"Do not provide additional description for any response including code functions."
        return prompt_content
        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m2(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m3(self,indiv1):
        prompt_content = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"+indiv1['code']+"\n"\
+self.prompt_inout_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_path_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+"\n"\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_path_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+"\n"\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Identify the common backbone idea in the provided algorithms." \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    
    def get_prompt_time(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved time by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_distance(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved distance by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_smoothness(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved smoothness by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_clearance(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved clearance by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_memory(self, indiv1):
        prompt_content = self.prompt_role+"\n"+self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please help us create a new algorithm with improved computing memory by modifying the provided algorithm. \n"\
"Identify the backbone idea in the provided algorithms." \
+ self.inherit_prompt + "\n" \
+ self.prompt_constraints + "\n" + self.prompt_constraints + "\n" + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content


    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        algorithm = algorithm_match.group(1).strip() if algorithm_match else ""

        code_match = re.search(r"\}\s*(.*)", response, re.DOTALL)
        # code_match = re.search(r"\}\s*[\s\S]*?(?=(import|class|from))((?:import|class|from)[\s\S]*)", response)
        code_match = re.search(r"\}\s*[\s\S]*?(?=(class PathPlanning))((?:class PathPlanning)[\s\S]*)", response)

        code = code_match.group(2).strip() if code_match else ""

        if code.endswith("```"):
            code = code[:-3].rstrip()

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
            algorithm = algorithm_match.group(1).strip() if algorithm_match else ""

            code_match = re.search(r"\}\s*(.*)", prompt_content, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
                    
            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm
        code = code

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]
    
    def _get_code(self, prompt_content):
        pass # TODO


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    
    def e1(self,parents):
      
        # prompt_content = self.get_prompt_e1(parents)
        prompt_content = self.get_prompt_path_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    
    def e2(self,parents):
        
        # prompt_content = self.get_prompt_e2(parents)
        prompt_content = self.get_prompt_path_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
    

    # 'time', 'distance', 'smoothness', 'clearance'

    def m_time(self, parents):
        prompt_content = self.get_prompt_time(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ time ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")

        return [code_all, algorithm]
    
    def m_distance(self, parents):
        prompt_content = self.get_prompt_distance(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ time ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")

        return [code_all, algorithm]
    
    def m_smoothness(self, parents):
        prompt_content = self.get_prompt_smoothness(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ time ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")

        return [code_all, algorithm]
    
    def m_clearance(self, parents):
        prompt_content = self.get_prompt_clearance(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ time ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")

        return [code_all, algorithm]
    
    def m_memory(self, parents):
        prompt_content = self.get_prompt_memory(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ time ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")

        return [code_all, algorithm]