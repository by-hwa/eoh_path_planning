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
import pandas as pd
import os

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba, output_path, n_op, **kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode
        self.output_path = output_path
        self.n_op = n_op

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.time_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "time_db.json")
        self.path_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "path_db.json")

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
    
    def population_generation_initial(self, data):
        population = []

        for i in range(len(data)):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                try:
                    future = executor.submit(self.interface_eval.evaluate, data[i]['code'])
                    fitness, results = future.result(timeout=self.timeout)

                    print(data[i]['algorithm'])
                    print(results)
                    
                    if fitness is None:
                        continue
                    seed_alg = {
                        'operator': 'initial',
                        'algorithm_description': data[i]['algorithm_description'],
                        'planning_mechanism': data[i]['planning_mechanism'],
                        'code': data[i]['code'],
                        'objective': fitness,
                        'other_inf': results.to_dict(orient='records') if isinstance(results, pd.DataFrame) else results
                    }
                except Exception as e:
                    print(f"Error in population_generation_initial: {traceback.format_exc()}")
                    seed_alg = {
                        'operator': 'initial',
                        'algorithm_description': data[i]['algorithm_description'],
                        'planning_mechanism': data[i]['planning_mechanism'],
                        'code': data[i]['code'],
                        'objective': None,
                        'other_inf': None
                    }
                    continue

            population.append(seed_alg)

        print("Initiliazation finished! Get "+str(len(population))+" Initial algorithms")

        return population

    def _get_alg(self,pop,operator):
        offspring = {
            'operator': operator,
            'algorithm_description': None,
            'planning_mechanism': None,
            'code': None,
            'objective': None,
            'time_improvement': None,
            'length_improvement': None,
            'other_inf': None,
        }
        if hasattr(self.evol, operator):
            if operator in ['cross_over', 'time_expert', 'path_expert']:
                parents = []

                if not operator == 'path_expert':
                    with open(file=self.time_db_path, mode='r') as f:
                        data = json.load(f)
                        if not len(data):
                            return None, None
                        parents.append(self.select.parent_selection(data, 1)[0])
                        
                if not operator == 'time_expert':
                    with open(file=self.path_db_path, mode='r') as f:
                        data = json.load(f)
                        if not len(data):
                            return None, None
                        parents.append(self.select.parent_selection(data, 1)[0])

            else:
                parents = self.select.parent_selection(pop, self.m)
            [offspring['code'], offspring['planning_mechanism'], offspring['algorithm_description']] = self.evol.evol(parents, operator)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring
    
    def get_offspring_code(self, pop, operator):
        try:
            p, offspring = self._get_alg(pop, operator)
            if offspring is None:
                return None, None

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator)
                    
                if n_retry > 1:
                    break

        except Exception as e:
            print(f"Error in get_offspring_code : {traceback.format_exc()}")
            offspring = {
            'operator': operator,
            'algorithm_description': None,
            'planning_mechanism': None,
            'code': None,
            'objective': None,
            'time_improvement': None,
            'length_improvement': None,
            'success_rate': None,
            'other_inf': None
            }
            p = None
        
        return p, offspring
            
    def get_offspring(self, pop, operator):
            
        p, offspring = self.get_offspring_code(pop, operator)
        if offspring is None:
            raise
        code = offspring['code'] if offspring['code'] else '' ### temporary

        with concurrent.futures.ThreadPoolExecutor() as executor:
            n_try = 0
            try:
                while n_try <= 3:
                    n_try+=1
                    future = executor.submit(self.interface_eval.evaluate, code)
                    fitness, results = future.result(timeout=self.timeout)

                    print('-----------------------------------')
                    print(fitness)
                    print(results)
                    print("***********************************")

                    offspring['objective'] = np.round(fitness, 5) if fitness is not None and not np.isnan(fitness) else None
                    offspring['other_inf'] = results.to_dict(orient='records') if isinstance(results, pd.DataFrame) else results

                    filename = self.output_path + "/results/pops/evaluated_entire_population_generation.json"
                    if offspring['objective']:
                        offspring['time_improvement'] = np.round(results['time_improvement'].mean())
                        offspring['length_improvement'] = np.round(results['length_improvement'].mean())
                        offspring['success_rate'] = np.round(results['success_rate'].mean())
                        with open(file=filename, mode='a') as f:
                            json.dump(offspring, f, indent=4)
                            f.write('\n')

                        if offspring['time_improvement'] > -800 and offspring['length_improvement'] > 18 and offspring['success_rate'] >= 1.0:
                            with open(file=self.time_db_path, mode='r+') as f:
                                data = json.load(f)
                                data.append(offspring)
                                
                                f.seek(0)
                                json.dump(data, f, indent=4)
                                f.truncate()

                        if offspring['time_improvement'] > 0 and offspring['length_improvement'] > 10 and offspring['success_rate'] >= 1.0:
                            with open(file=self.path_db_path, mode='r+') as f:
                                data = json.load(f)
                                data.append(offspring)

                                f.seek(0)
                                json.dump(data, f, indent=4)
                                f.truncate()
                        break

                    elif offspring['other_inf'] is None:
                        print(f"Try new code generation : {n_try}")
                        p, offspring = self.get_offspring_code(pop, operator)
                        code = offspring['code']

                    elif 'Traceback' in offspring['other_inf']:
                        print(f"Error in ThreadPoolExecutor : {offspring['other_inf']['Traceback']}")
                        filename = self.output_path + "/results/pops/error_occured_entire_population_generation.json"
                        with open(file=filename, mode='a') as f:
                            json.dump(offspring, f, indent=4)
                            f.write('\n')
                        print(f'Trying Trouble shoot {n_try}')
                        code = self.evol.trouble_shoot(code, offspring['other_inf']['Traceback'])
                        offspring['code'] = code
                        print('Trouble shooted CODE')
                        print(code)
                        continue

            except Exception as e:
                print(f"Error in get_offspring: {traceback.format_exc()}")
                offspring['objective'] = None
                offspring['results'] = None
                time.sleep(1)
                
            future.cancel()

        print(offspring['objective'])

        # Round the objective values
        return p, offspring

    def get_algorithm(self, pop, operators):
        # operators = ['cross_over', 'cross_over', 'cross_over']
        results = []
        try:
            for operator in operators:
                p, off = self.get_offspring(pop, operator)
                if off['objective'] is not None:
                    results.append((p, off))
                    print("!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    print(f"Offspring with operator {operator} has no valid objective, skipping...")

        except Exception as e:
            if self.debug:
                print(f"Error in get_algorithm: {traceback.format_exc()}")
            
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

