import glob
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
from typing import Dict, Optional, Any, List, NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd

""" Relevant Directories """

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

"""Available GPUs"""


class GPU(NamedTuple):
    name: str
    gpu_memory: int  # In GBs


available_gpus = {
    0: GPU(name='gtx_1080_ti', gpu_memory=11),
    1: GPU(name='rtx_2080_ti', gpu_memory=11),
    2: GPU(name='rtx_3090', gpu_memory=24),
    3: GPU(name='rtx_4090', gpu_memory=24),
    4: GPU(name='titan_rtx', gpu_memory=24),
    5: GPU(name='quadro_rtx_6000', gpu_memory=24),
    6: GPU(name='v100', gpu_memory=32),
    7: GPU(name='a100-pcie-40gb', gpu_memory=40),
    8: GPU(name='a100_80gb', gpu_memory=80),
}

""" Async executor """


class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def _dummy_fun():
    pass


""" Command generators """


def generate_base_command(module, flags: Optional[Dict[str, Any]] = None, unbuffered: bool = True) -> str:
    """ Generates the command to execute python module with provided flags

    Args:
        module: python module / file to run
        flags: dictionary of flag names and the values to assign to them.
               assumes that boolean flags are encoded as store_true flags with False as default.
        unbuffered: whether to invoke an unbuffered python output stream

    Returns: (str) command which can be executed via bash

    """

    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    if unbuffered:
        base_cmd = interpreter_script + ' -u ' + base_exp_script
    else:
        base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool or type(setting) == np.bool_:
                if setting:
                    base_cmd += f" --{flag}"
            else:
                base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list: List[str],
                          output_file_list: Optional[List[str]] = None,
                          num_cpus: int = 1,
                          num_gpus: int = 0,
                          gpu: GPU | None = None,
                          dry: bool = False,
                          mem: int = 2 * 1028,
                          duration: str = '3:59:00',
                          mode: str = 'local',
                          prompt: bool = True) -> None:
    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'sbatch ' + \
                   f'--time={duration} ' + \
                   f'--mem-per-cpu={mem} ' + \
                   f'--cpus-per-task {num_cpus} '

        if num_gpus > 0 and gpu is not None:
            bsub_cmd += f'--gpus={gpu.name}:{num_gpus} '
        elif num_gpus > 0:
            bsub_cmd += f'-G {num_gpus} --gres=gpumem:10240m '

        assert output_file_list is None or len(command_list) == len(output_file_list)
        if output_file_list is None:
            for cmd in command_list:
                cluster_cmds.append(bsub_cmd + f'--wrap="{cmd}"')
        else:
            for cmd, output_file in zip(command_list, output_file_list):
                cluster_cmds.append(bsub_cmd + f'--output={output_file} --wrap="{cmd}"')

        if dry:
            for cmd in cluster_cmds:
                print(cmd)
        else:
            if prompt:
                answer = input(f"about to launch {len(command_list)} jobs with {num_cpus} "
                               f"cores each. proceed? [yes/no]")
            else:
                answer = 'yes'
            if answer == 'yes':
                for cmd in cluster_cmds:
                    os.system(cmd)

    elif mode == 'local':
        if prompt:
            answer = input(f"about to run {len(command_list)} jobs in a loop. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if prompt:
            answer = input(f"about to launch {len(command_list)} commands in {num_cpus} "
                           f"local processes. proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                executor = AsyncExecutor(n_jobs=num_cpus)
                executor.run(lambda command: os.system(command), command_list)
    else:
        raise NotImplementedError


""" Some aggregation functions """


def dict_permutations(d: Optional[dict]) -> List[Dict]:
    if not d:
        return []
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary.")
    
    keys = d.keys()
    values = d.values()
    perms = []

    # Calculate the Cartesian product of all values in the dictionary
    for value_combo in itertools.product(*values):
        perms.append(dict(zip(keys, value_combo)))

    return perms


if __name__ == '__main__':
    # Example for dict_permutations
    d = {
        "A": [1, 2],
        "B": ["x", "y"],
        "C": ["!", "@"]
    }

    result = dict_permutations(d) + dict_permutations(None) + dict_permutations({})
    for r in result:
        print(r)

    try:
        print(dict_permutations(123))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(dict_permutations("not_a_dict"))
    except ValueError as e:
        print(f"Error: {e}")
