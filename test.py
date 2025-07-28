import sys
sys.path.append('C:\Workspace\PathBench\src')
import multiprocessing
from structures import Point  # 문제의 Point class

def worker(p: Point):
    print(p)
    print(p.is_float)  # <- 여기서 에러 나면 multiprocessing 문제일 가능성 높음

if __name__ == '__main__':
    p = Point(1, 2)
    proc = multiprocessing.Process(target=worker, args=(p,))
    proc.start()
    proc.join()
