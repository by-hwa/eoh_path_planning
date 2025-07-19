import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import traceback
import json
import sys
import math
class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba, output_path, hier_gen, n_op, **kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        prompts.import_string = self.interface_eval.get_import_string()
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, hier_gen=hier_gen, **kwargs)
        self.m = m
        self.debug = debug_mode
        self.output_path = output_path
        self.hier_gen = hier_gen
        self.n_op = n_op

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba

    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: InterfaceEC.convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [InterfaceEC.convert_numpy(i) for i in obj]
        return obj
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation(self):
        
        n_create = 2
        
        population = []

        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1')
            for p in pop:
                population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population

    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if hasattr(self.evol, operator):
            parents = self.select.parent_selection(pop,self.m, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.evol(parents, operator)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring
    

    def get_offspring(self, pop, operator):

        try:
            p, offspring = self._get_alg(pop, operator)
            code = offspring['code']

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator)
                code = offspring['code']
                    
                if n_retry > 1:
                    break

            print("here is the offspring code: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(code)

            filename = self.output_path + "/results/pops/entire_population_generation.json"
            with open(file=filename, mode='a') as f:
                json.dump(offspring, f, indent=5)
                f.write('\n')

                            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                n_try = 0
                while n_try <= 3:
                    n_try += 1
                    try:
                        future = executor.submit(self.interface_eval.evaluate, code)
                        fitness, results = future.result(timeout=self.timeout)

                        print('-----------------------------------')
                        print(fitness)
                        print(results)
                        print("***********************************")

                        offspring['objective'] = np.round(fitness, 5) if fitness else None
                        offspring['results'] = results
                        offspring['results'] = {k: v for k, v in offspring['results'].items() if "alldata" not in k}
                        
                        filename = self.output_path + "/results/pops/evaluated_entire_population_generation.json"
                        if offspring['objective'] and offspring['objective'] != float("inf"):
                            with open(file=filename, mode='a') as f:
                                json.dump(self.convert_numpy(offspring), f, indent=5)
                                f.write('\n')
                            break
                        elif offspring['objective'] == float("inf"):
                            break

                        
                    except Exception as e:
                        print(f"Error in ThreadPoolExecutor : {offspring['results']['Traceback']}")
                        if n_try <= 3:
                            print(f'Trying Trouble shoot {n_try}')
                            code = self.evol.trouble_shoot(code, offspring['results']['Traceback'])
                            offspring['code'] = code
                            print('Trouble shooted CODE')
                            print(code)
                    
                future.cancel()


        except Exception as e:
            print(f"Error in get_offspring : {traceback.format_exc()}")
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            print(offspring)
            p = None

        print(offspring['objective'])

        # Round the objective values
        return p, offspring

    def get_algorithm(self, pop, operators):
        results = []
        try:
            results = Parallel(n_jobs=self.n_p,prefer='threads', timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for operator in operators)
        except Exception as e:
            if self.debug:
                print(f"Error in get_algorithm: {traceback.format_exc()}")
            print("Parallel time out .")
            
        time.sleep(2)
        print(len(results))
        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check population: \n {p}")
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

