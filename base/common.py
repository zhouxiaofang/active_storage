import time
from functools import wraps
import psutil

def fun_run_time(func):
    ''' Used to calculate the running time of decorated function. ''' 
    @wraps(func)
    def inner(*args, **kwargs):
        s_time = time.time()
        ret = func(*args, **kwargs)
        e_time = time.time()
        print('{} cost {} s'.format(func.__name__, e_time - s_time))
        return ret
    
    return inner


def print_memory_info(str_output):
    mem = psutil.virtual_memory()
    # 系统总计内存
    zj = float(mem.total) / 1024 / 1024
    # 系统已经使用内存
    ysy = float(mem.used) / 1024 / 1024 
    # 系统空闲内存
    kx = float(mem.free) / 1024 / 1024

    print('系统-{0}-总计内存:{1} M.'.format(str_output, zj))
    print('系统-{0}-已经使用内存:{1} M.'.format(str_output, ysy))
    print('系统-{0}-空闲内存:{1} M.'.format(str_output, kx))
