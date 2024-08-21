# model-based-rl
Repository for doing model based RL

## Getting started

### Local Installation

1. Requirements:  
    - Python >=3.11
    - CUDA >= 12.1
    - cudnn >= 8.9

3. Install [JAX](https://jax.readthedocs.io/en/latest/installation.html) either on CPU or GPU:
    ```shell
    pip install -U "jax[cpu]"
    pip install -U "jax[cuda12]"
    ```

2. Install with a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
    ```shell
    conda create -n mbrl python=3.11 -y
    conda activate mbrl
    git clone https://github.com/lasgroup/model-based-rl.git
    pip install .
    ```

4. set up [wandb](https://docs.wandb.ai/quickstart)

5. add mbrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/to/model-based-rl```. You can also add this to your .bashrc.

6. Launch experiments with the launcher: 
    ```
    python path/to/model-based-rl/experiments/experiment_name/launcher.py
    ```


### Remote Deployment on [euler.ethz.ch](https://scicomp.ethz.ch/wiki/Main_Page)

1. Set up remote development from your computer to Euler in either [PyCharm](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#mapping) ~~or [VSCode](https://code.visualstudio.com/docs/remote/ssh-tutorial)~~.

5. Set up git protocols on Euler: [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

2. Set up a .mbrl_setup file on your login node:
    ```shell
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_DETERMINISTIC_OPS=0

    module load stack/2024-06
    module load gcc/12.2.0
    module load eth_proxy
    module load python/3.11.6

    PYTHONPATH=$PYTHONPATH:/cluster/home/kiten/copax/model-based-rl
    export PYTHONPATH
    ```
    Source it with `source .mbrl_setup`.

3. Create a [miniconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [python virtual environment](https://docs.python.org/3.11/library/venv.html#creating-virtual-environments).

5. activate virtual environment:
    ```shell
    source path/on/euler/to/venv/bin/activate
    ```

5. Install Jax for GPU (see [the JAX documentation](https://jax.readthedocs.io/en/latest/installation.html))
    ```shell
    pip install "jax[cuda12]"
    ```

4. git clone and pip install the mbrl library:
    ```shell
    git clone https://github.com/lasgroup/model-based-rl.git
    pip install .
    ```

5. set up [wandb](https://docs.wandb.ai/quickstart) on Euler

5. add mbrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/on/euler/to/model-based-rl```. You can also add this to your .bashrc or .mbrl_setup file.

6. Launch experiments with the launcher: 
    ```
    python path/on/euler/to/model-based-rl/experiments/experiment_name/launcher.py
    ```

### Error Messages

#### Pip dependency errors when installing/updating mbrl:
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```
This should generally not be a problem. Be sure to use the most up-to-date versions of jax, chex, flax.

Here's a combination that works on CentOS:
 ```
brax==0.10.0
chex>=0.1.82
flax==0.7.2
jax==0.4.14
jaxlib==0.4.14
jaxopt==0.8
jaxtyping==0.2.21
jaxutils==0.0.8
ml-collections==0.1.0
numba
numpy>=1.25.2
scipy==1.11.2
setuptools>=68.1.2
tensorflow>=2.13.0
```

#### CPU/GPU error
```
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
about to launch 5 jobs with 1 cores each. proceed? [yes/no]
```
This is a normal warning when launching from the login node on Euler, since GPUs are only available on the computing node. It should use the GPU once the job is submitted.


#### Segmentation Fault Error on Euler while Dynamics Training or Policy Evaluation
No fix found yet.
