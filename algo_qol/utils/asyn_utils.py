# -*- coding: utf-8 -*-
"""
@Time ： 2024/09/23 14:13
@Auth ： heshuai.sec@gmail.com
"""
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


class AsyncExecutor:
    def __init__(self, qps: int = 100, max_workers: int = None):
        self.qps = qps
        self.max_workers = max_workers or qps
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.semaphore = threading.Semaphore(self.qps)
        self.tasks = []

    def submit(self, func, *args, **kwargs):
        with self.semaphore:
            future = self.executor.submit(func, *args, **kwargs)
            self.tasks.append(future)
            return id(future)

    def get_result(self, task_id):
        for future in self.tasks:
            if id(future) == task_id:
                return future.result()
        raise ValueError(f"Task with ID {task_id} not found or completed.")

    def get_all_results(self):
        results = []
        for future in as_completed(self.tasks):
            results.append(future.result())
        self.tasks.clear()  # Clear the list after retrieving results
        return results

    def close(self):
        self.executor.shutdown(wait=True)
