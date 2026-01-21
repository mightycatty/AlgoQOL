"""
@Time ： 2024/10/10 10:45
@Auth ： heshuai.sec@gmail.com
@File ：utils.py
"""
import time
import logging


class Timer:

    def __init__(self):
        self.timers = {}
        self._durations = {}

    def start(self, name):
        if name in self.timers:
            logging.warning(f"Timer '{name}' already started.")
        self.timers[name] = time.time()

    def end(self, name, verbose=False):
        if name not in self.timers:
            logging.warning(f"Timer '{name}' not started.")
            return 0
        end_time = time.time()
        run_time = end_time - self.timers[name]
        del self.timers[name]
        if verbose:
            print(f"{name} runtime: {run_time}")
        self._durations[name] = run_time
        return run_time

    def duration(self, name):
        if name not in self._durations:
            logging.warning(f"Timer '{name}' not started.")
            return 0
        else:
            return self._durations[name]

    def reset(self):
        self.timers = {}
        self._durations = {}
