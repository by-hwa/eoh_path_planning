import pandas as pd
import numpy as np
import random
import time
from typing import Tuple, Literal, Union, Optional, List, Dict, NamedTuple, Callable, Any
import plotly.express as px
from .architecture_utils import PlannerResult
import math
# from eoh.problems.optimization.classic_benchmark_path_planning.utils.architecture_utils import PlannerResult
class MultiMapBenchmarker:
    def __init__(
        self,
        maps: List[np.ndarray],
        name: str = "Algorithm",
        learning_mode = True,
        iter = 10,
        seed = 42,
    ):
        self.set_seed(seed)

        self.maps = maps
        self.name = name
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.iter = iter
        self.learning_mode = learning_mode

        self.time_limit = 5.0
        self.success_limit = 0.8

    def run(self, algorithm) -> pd.DataFrame:
        results = []
        main_start_time = time.time()
        for i, map_ in enumerate(self.maps):
            print(f"Map {i+1}")
            for j in range(self.iter):
                start_time = time.time()
                try:
                    output = algorithm(map_)
                except Exception as e:
                    output = {"path": [], "nodes": [], "n_nodes": 0}
                end_time = time.time()


                if isinstance(output, PlannerResult):
                    path = output.path
                    nodes = output.nodes
                    num_nodes = len(nodes)
                    success = self._is_path_valid(path, map_)
                    path_length = self._compute_path_length(path)
                elif isinstance(output, Dict):
                    path = output['path']
                    nodes = output['nodes']
                    num_nodes = len(nodes)
                    success = self._is_path_valid(path, map_)
                    path_length = self._compute_path_length(path)
                else:
                    return None, None
                
                print(f"Iteration {j+1}: Time taken: {end_time - start_time:.4f} seconds, Success: {success}")

                results.append({
                    "map_id": i,
                    "iter" : j,
                    "algorithm": self.name,
                    "success": success,
                    "time_taken": end_time - start_time,
                    "num_nodes": num_nodes,
                    "path_length": path_length
                })

                time_taken_avg = np.mean([r["time_taken"] for r in results])
                success_avg = np.mean([r["success"] for r in results]) if len(results) > 6 else 1.0
                if time_taken_avg > self.time_limit:
                    print("Time taken Limit")
                    return None, None
                elif success_avg < self.success_limit:
                    print("Success Rate Limit")
                    return None, None

        print(f"Total time taken for all maps: {time.time() - main_start_time:.4f} seconds")

        self.results_df = pd.DataFrame(results)
        return self.results_df, self.get_avg()

    def _compute_path_length(self, path: List[Tuple[float, ...]]) -> float:
        if not path or len(path) < 2:
            return 0.0
        return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path) - 1))
    
    def _is_path_valid(self, path: List[Tuple[float, ...]], map_) -> bool:
        if not path or len(path) < 2:
            return False
        
        if np.linalg.norm(np.array(path[0]) - np.array(map_.start)) > 0.5 or np.linalg.norm(np.array(path[-1]) - np.array(map_.goal)) > 0.5:
            if np.linalg.norm(np.array(path[0]) - np.array(map_.goal)) > 0.5 or np.linalg.norm(np.array(path[-1]) - np.array(map_.start)) > 0.5:
                return False
            
        is_3d = True if len(map_.size) > 2 else False
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if self._is_edge_in_obstacle(p1, p2, map_.obstacles, is_3d):
                return False
        return True
    
    def _is_in_obstacle(self, pos, obstacles, is_3d):
        for obs in obstacles:
            if is_3d:
                x, y, z, w, h, d = obs
                px, py, pz = pos
                if x <= px <= x + w and y <= py <= y + h and z <= pz <= z + d:
                    return True
            else:
                x, y, w, h = obs
                px, py = pos
                if x <= px <= x + w and y <= py <= y + h:
                    return True
        return False

    def _is_edge_in_obstacle(self, from_pos, to_pos, obstacles, is_3d, resolution=1.0):
        distance = math.dist(from_pos, to_pos)
        steps = max(1, int(distance / resolution))
        for i in range(steps + 1):
            interp = tuple(from_pos[d] + (to_pos[d] - from_pos[d]) * (i / steps) for d in range(len(from_pos)))
            if self._is_in_obstacle(interp, obstacles, is_3d):
                return True
        return False

    def save_results(self, filename: str):
        if self.results_df.empty:
            raise RuntimeError("No results to save. Run benchmark first.")
        self.results_df.to_csv(filename, index=False)

    def plot_metrics(self, metric: str = "time_taken"):
        if self.results_df.empty:
            raise RuntimeError("No results to plot. Run benchmark first.")
        if metric not in self.results_df.columns:
            raise ValueError(f"Invalid metric: {metric}")

        fig = px.bar(
            self.results_df,
            x="map_id",
            y=metric,
            color="success",
            title=f"{self.name} - {metric} per map",
            labels={"map_id": "Map ID", metric: metric.replace('_', ' ').title()}
        )
        fig.show()

    def get_avg(self):
        if self.results_df.empty:
            raise RuntimeError("No results to save. Run benchmark first.")

        classic_summary = self.results_df.groupby('map_id').agg({
            'success': 'mean',           # 성공률
            'time_taken': 'mean',        # 평균 소요 시간
            'num_nodes': 'mean',      # 평균 노드 수
            'path_length': lambda x: x[x > 0].mean()  # path=0 제외한 평균 경로 길이
        }).rename(columns={
            'success': 'success_rate',
            'time_taken': 'time_avg',
            'num_nodes': 'num_nodes_avg',
            'path_length': 'path_length_avg'
        }).reset_index()
        return classic_summary
    
    @staticmethod
    def get_improvement(reference_result:pd.DataFrame, results:pd.DataFrame) -> pd.DataFrame:
        improvement_df = pd.DataFrame()
        
        improvement_df['success_improvement'] = (results['success_rate'] - reference_result['success_rate']) * 100 # percent point
        improvement_df['time_improvement'] = (results['time_avg'] - reference_result['time_avg']) / reference_result['time_avg'] * -100
        improvement_df['length_improvement'] = (results['path_length_avg'] - reference_result['path_length_avg']) / reference_result['path_length_avg'] * -100

        improvement_df['objective_score'] = (
            5 * improvement_df['success_improvement'] +
            0.3 * improvement_df['time_improvement'] +
            0.2 * improvement_df['length_improvement']
            )

        return improvement_df
    
    def set_seed(self, seed: int = 42):
        random.seed(seed)                    # Python random
        np.random.seed(seed)                 # NumPy
        # torch.manual_seed(seed)              # PyTorch (CPU)
        # torch.cuda.manual_seed(seed)         # PyTorch (GPU)
        # torch.cuda.manual_seed_all(seed)     # If multiple GPUs
        # torch.backends.cudnn.deterministic = True  # CUDNN 고정
        # torch.backends.cudnn.benchmark = False     # 연산 최적화 OFF