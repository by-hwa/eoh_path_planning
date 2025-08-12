import numpy as np
import time
import multiprocessing as mp
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
        self.smoothness_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "smoothness_db.json")
        self.time_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "time_analysis_db.json")
        self.path_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "path_analysis_db.json")
        self.smoothness_analysis_db_path = os.path.join(self.base_dir, "..", "..", "problems", "optimization", "classic_benchmark_path_planning", "utils", "database", "smoothness_analysis_db.json")

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
    
    def _compute_improvement(self, parent, child, metric, maximize=False):
        if maximize:
            return (child - parent) / parent
        else:
            return (parent - child) / parent
    
    def _compute_adjust_score(self, parents, child, metric):
        metric_name = metric+'_improvement'
        best_parent = None
        best_val = None
        for p in parents:
            if best_val is None or p[metric_name] > best_val:
                best_val = p[metric_name]
                best_parent = p

        if best_parent is None:
            return 0

        improvement = best_parent.get(metric + '_improvement', 0) - child.get(metric + '_improvement', 0)
        final_perform = child.get(metric + '_improvement', 0)

        return improvement * 0.4 + final_perform * 0.6
    
    def _save_data(self, file_path, contents):
        with open(file=file_path, mode='r+') as f:
            data = json.load(f)
            data.append(contents)
            
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        print(f"[{time.strftime('%Y.%m.%d - %H:%M:%S')}] {file_path} DB SAVED -----------------")

    def is_improvement(self, parents, offspring, metric):
        if not parents: return False
        for p in parents:
            if not metric in p.keys():
                continue
            elif p[metric] > offspring[metric]:
                False
        return True
                            
    def save_analysis_db(self, p, offspring, metric, save_path):
        parents_codes = [inv['code'] for inv in p]
        analysis = self.evol.get_analysis(parents_codes, offspring['code'], metric)
        contents = {
            'parents': parents_codes,
            'offspring': offspring,
            'objective': self._compute_adjust_score(p, offspring, metric),
            'analysis': analysis
            }
        self._save_data(save_path, contents)

        
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
                        'time_improvement': np.round(results['time_improvement'].mean()),
                        'length_improvement': np.round(results['length_improvement'].mean()),
                        'smoothness_improvement': np.round(results['smoothness_improvement'].mean()),
                        'other_inf': results.to_dict(orient='records') if isinstance(results, pd.DataFrame) else results
                    }
                    seed_alg['success_rate'] = results['success_rate'].mean()
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
            'smoothness_improvement': None,
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
            'smoothness_improvement': None,
            'success_rate': None,
            'other_inf': None
            }
            p = None
        
        return p, offspring


    def get_offspring(self, pop, operator):
        p, offspring = self.get_offspring_code(pop, operator)
        if offspring is None:
            raise RuntimeError("get_offspring_code returned None")

        code = offspring.get('code', '')  # temporary
        n_try = 0
        while n_try <= 3:
            n_try += 1
            # === run evaluate with HARD timeout ===
            with concurrent.futures.ThreadPoolExecutor() as executor:
                try:
                    if code is None:code =''
                    future = executor.submit(self.interface_eval.evaluate, code)
                    fitness, results = future.result(timeout=self.timeout)
                except:
                    fitness, results = None, None

                future.cancel()

            print('-----------------------------------')
            print(fitness)
            print(results)
            print("***********************************")

            # Normalize outputs
            is_df = isinstance(results, pd.DataFrame)
            offspring['objective'] = np.round(fitness, 5) if (fitness is not None and not np.isnan(fitness)) else None
            offspring['other_inf'] = results.to_dict(orient='records') if is_df else results

            # Early retry paths
            if offspring['objective'] is None:
                # if results is None or (isinstance(results, dict) and 'Traceback' in results):
                    # Error or timeout: try troubleshoot if we have a traceback; otherwise regenerate code
                    # if isinstance(results, dict) and 'Traceback' in results:
                    #     print(f"Error/Timeout: {results['Traceback']}")
                    #     filename = self.output_path + "/results/pops/error_occured_entire_population_generation.json"
                    #     with open(file=filename, mode='a') as f:
                    #         json.dump(offspring, f, indent=4)
                    #         f.write('\n')

                    #     # Try troubleshooting – keep same offspring but patch code
                    #     try:
                    #         code = self.evol.trouble_shoot(code, results['Traceback'])
                    #         offspring['code'] = code
                    #         print('Troubleshot CODE')
                    #         print(code)
                    #     except Exception:
                    #         # If troubleshooting itself fails, fall back to new code
                    #         print('Troubleshoot failed; regenerating code')
                    #         p, offspring = self.get_offspring_code(pop, operator)
                    #         code = offspring['code']
                    # else:
                print(f"Try new code generation : {n_try}")
                p, offspring = self.get_offspring_code(pop, operator)
                code = offspring['code']
                    # continue
                # else:
                #     # Some non-error but no objective -> stop
                #     offspring['results'] = None
                #     break

            # === objective exists: compute aggregated metrics (guarded) ===
            filename = self.output_path + "/results/pops/evaluated_entire_population_generation.json"

            offspring['time_improvement'] = np.round(results['time_improvement'].mean())
            offspring['length_improvement'] = np.round(results['length_improvement'].mean())
            offspring['smoothness_improvement'] = np.round(results['smoothness_improvement'].mean())
            offspring['success_rate'] = results['success_rate'].mean()

            # Persist evaluation snapshot
            with open(file=filename, mode='a') as f:
                json.dump(offspring, f, indent=4)
                f.write('\n')

            # === gating and DB updates (unchanged logic, but with None guards) ===
            ti, li, si, sr = (offspring['time_improvement'], offspring['length_improvement'],
                            offspring['smoothness_improvement'], offspring['success_rate'])

            if (ti is not None and li is not None and sr is not None
                and ti > -800 and li > 20 and sr >= 1.0):
                self._save_data(self.path_db_path, offspring)
                if self.is_improvement(p, offspring, 'length_improvement'):
                    self.save_analysis_db(p, offspring, 'length', self.path_analysis_db_path)

            if (ti is not None and li is not None and sr is not None
                and ti > 50 and li > 10 and sr >= 1.0):
                self._save_data(self.time_db_path, offspring)
                if self.is_improvement(p, offspring, 'time_improvement'):
                    self.save_analysis_db(p, offspring, 'time', self.time_analysis_db_path)

            if (ti is not None and li is not None and si is not None and sr is not None
                and ti > -500 and li > 20 and si > 1000 and sr >= 1.0):
                self._save_data(self.smoothness_db_path, offspring)
                if self.is_improvement(p, offspring, 'smoothness_improvement'):
                    self.save_analysis_db(p, offspring, 'smoothness', self.smoothness_analysis_db_path)

            break  # success path: exit retry loop

        print(offspring['objective'])
        return p, offspring

    def get_algorithm(self, pop, operators):
        # operators = ['cross_over', 'cross_over', 'cross_over']
        self.evol.load_analysis()
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

