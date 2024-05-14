# model-based-rl
Repository for doing model based RL

## Getting started

### Local Installation

1. Requirements:  
    - Python >=3.11
    - CUDA >= 12.1
    - cudnn >= 8.9

2. Installation
    ```shell
    conda create -n mbrl python=3.11 -y
    conda activate mbrl
    git clone https://github.com/sukhijab/model-based-rl.git
    pip install .
    ```
3. Install [JAX](https://jax.readthedocs.io/en/latest/installation.html) either on CPU or GPU:
    ```shell
    pip install -U "jax[cpu]"
    pip install -U "jax[cuda12]"
    ```

### Remote Deployment on [euler.ethz.ch](https://scicomp.ethz.ch/wiki/Main_Page)

1. Set up remote development in either [PyCharm](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#mapping) ~~or [VSCode](https://code.visualstudio.com/docs/remote/ssh-tutorial)~~.

5. Set up git protocols: [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

2. Set up a ._setup file on your login node:
    ```shell
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_DETERMINISTIC_OPS=0
    env2lmod
    module load gcc/8.2.0
    module load python/3.11.2
    module load cuda/12.1.1
    module load cudnn/8.9.2.26
    module load eth_proxy
    ```
    Source it with `source ._setup`.

3. Create a [miniconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [python virtual environment](https://docs.python.org/3.11/library/venv.html#creating-virtual-environments).

4. git clone and pip install the mbrl library:
    ```shell
    git clone https://github.com/sukhijab/model-based-rl.git
    pip install .
    ```

5. You might have to install a specific jax library version (see [the JAX documentation](https://jax.readthedocs.io/en/latest/installation.html))
    ```shell
    pip install jaxlib==0.4.14+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

5. set up wandb: https://docs.wandb.ai/quickstart

6. get it to run
    - Check envs --> State Class --> Continuous State??? 

7. in base_model_based_agent.py: Look around.
    system_wrapper.py: make it CT
    Overwrite environment (Brax does not give derivatives )--> write your own or chec what Lenart did with OCORL.
    or adjust BRAX to add wrapper for ODE and add derivatives to brax base.
    In wtc envs: Pendulum already has ODEs.

### Error Messages

#### Pip dependency errors:
You fucked up. Reconsider your environment.
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
numba 0.57.0 requires numpy<1.25,>=1.21, but you have numpy 1.26.4 which is incompatible.
tensorflow 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 1.26.4 which is incompatible.
scvi-tools 1.0.2 requires chex<=0.1.8, but you have chex 0.1.86 which is incompatible.
scvi-tools 1.0.2 requires ml-collections>=0.1.1, but you have ml-collections 0.1.0 which is incompatible.
pymrio 0.5.4 requires openpyxl<3.1.1,>=3.0.6, but you have openpyxl 3.1.2 which is incompatible.
jaxutils 0.0.8 has requirement ml-collections==0.1.0, but you have ml-collections 0.1.1.
cvxpy 1.3.1 has requirement setuptools>65.5.1, but you have setuptools 65.5.0.
pymrio 0.5.4 has requirement openpyxl<3.1.1,>=3.0.6, but you have openpyxl 3.1.2.
```
Here's a combination that works, probably:
 
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

#### Random CPU/GPU error
Fix: TODO
```
warnings.warn(f"Jitted function has {argnums_name}={argnums}, "
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
about to launch 5 jobs with 1 cores each. proceed? [yes/no]no
```
