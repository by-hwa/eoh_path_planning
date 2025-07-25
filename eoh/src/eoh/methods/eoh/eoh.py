import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC
from .classic_planning_method import GetPlanningCode
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator
        self.get_initial = paras.get_initial
        self.ref_algorithm = paras.ref_algorithm if self.get_initial else None
        self.hier_gen = paras.hier_gen

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    

    # run eoh 
    def run(self):

        print("- Evolution Start -")
        time_start = time.time()

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba, output_path=self.output_path, hier_gen=self.hier_gen, n_op=len(self.operators)
                                   )
        # write astar result
        filename = self.output_path + "/results/pops/evaluated_entire_population_generation.json"
        with open(file=filename, mode='a') as f:
            f.write('Astar')
            filtered = {k: v for k, v in interface_prob.ref_statistics.items() if "alldata" not in k}
            json.dump(InterfaceEC.convert_numpy(filtered), f, indent=5)
            f.write('\n')

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data,self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        elif self.get_initial:
            get_planning_code = GetPlanningCode()
            for algorithm in self.ref_algorithm:
                offspring = {
                'algorithm': get_planning_code.get_algorithm_description(algorithm),
                'code': get_planning_code.get_code(algorithm),
                'objective': 0,
                'other_inf': None
                }
                population.append(offspring)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            print("initial population has been Loaded!")
            n_start = 0

        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                print(f'population len {len(population)}')
                population = self.manage.population_management(population, self.pop_size)
                print(f"------------pop size{len(population)}-------------------")
                
                print(f"Pop initial: ")
                for off in population:
                    print('------------------')
                    print(" Obj: ", off['objective'], end="|")
                    print("check - population generate")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0
        

        # main loop
        n_op = len(self.operators)
        filename = self.output_path + "/results/pops/entire_population_generation.json"

        for pop in range(n_start, self.n_pop):  
            for i in range(self.pop_size//n_op):
                # self.operators = ["e1"] # for debug
                parents, offsprings = interface_ec.get_algorithm(population, self.operators)
                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring


            for off in offsprings:
                print(" Obj: ", off['objective'], end="|")
                print(f"\n len pop size----------------{len(population)}")
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

                # time.sleep(100000)


            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(InterfaceEC.convert_numpy(population), f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            print(f"population size in eoh.py : {len(population)}")
            with open(filename, 'w') as f:
                json.dump(InterfaceEC.convert_numpy(population[0]), f, indent=5)


            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()

