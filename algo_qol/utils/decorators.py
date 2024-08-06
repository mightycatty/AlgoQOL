import logging
import os
import time
from functools import wraps
from multiprocessing import Pool

import cloudpickle


def line_profile(func):
    """
    function decoderator for line-wise profile
    usage:
        from evaluation_utils import line_profile
        @line_profile
        some_fn()
    reference:
        https://github.com/rkern/line_profiler#kernprof
    :param func:
    :return:
    """
    import line_profiler
    prof = line_profiler.LineProfiler()

    @wraps(func)
    def newfunc(*args, **kwargs):
        try:
            pfunc = prof(func)
            return pfunc(*args, **kwargs)
        finally:
            prof.print_stats(1e-3)

    return newfunc


def timethis(func):
    '''
    Decorator that reports the execution time.
    '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result

    return wrapper

