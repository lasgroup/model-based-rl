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

6. TODOs and Notes
    - Check envs --> State Class --> Continuous State --> Change state class from brax
        - Maybe store the derivative in the pipeline state in env. That's basically where you dump all the unwanted stuff. So it doesn't fuck up Brax
    - In wtc envs: Pendulum already has ODEs.
        - Real talk: What does wtc do? It also has envs? What is the difference to the envs provided in mbrl? Is wtc in CT already? 
        - wtc provides some random matrix stuff? Huh? Why is half of it in mbrl.model_based_agent and not in wtc repo?
        - Lenart: You don't need wtc, it's for something else
    - Start in exp.py and go through it to understand it.
    - in base_model_based_agent.py: Look around.
    - SAC Optimizer is a discrete optimizer. It basically will predict the derivative so in system_wrapper (PetsDynamics), you can multiply the output of the optimizer with dt.
    - system_wrapper.py: make it CT
    - Overwrite environment (Brax does not give derivatives )--> write your own or check what Lenart did with OCORL.
        - or adjust BRAX to add wrapper for ODE and add derivatives to brax base.
        - Check out brax, what does it do in pendulum env, system wrapper, exp?
        - CHECK WITH JAX
        - Maybe just need to change PETS Dynamics and PETS System can stay the same. Same with Optimistic case.

    - Readings (required?):
        - Read about pets
            - Most naive thing: Just predict next state with NN. Esentially the mean of the ensemble and ignore the epistemic uncertainty. 
            - Next best thing: Take the epistemic uncertainty as noise. Sample from that.
            - Third is optimism. We have guarantees for that. 
            - Basically do all of that.
            - There is a dt in Brax. Make it smol. The environment dt and the integration dt should be the same.
        - Read about reparametrization trick in OCORL
            - Not really needed, since we use optimistic dynamics with small dt.


### Error Messages

#### Pip dependency errors:
You fucked up. Reconsider your environment.
```bash
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
```bash
warnings.warn(f"Jitted function has {argnums_name}={argnums}, "
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
about to launch 5 jobs with 1 cores each. proceed? [yes/no]no
```

### The current error
```bash
(venv) [kiten@eu-login-14 optimistic_dynamics]$ sbatch --gpus=1 --wrap="nvidia-smi"
Submitted batch job 58648651
(venv) [kiten@eu-login-14 optimistic_dynamics]$ cat slurm-58648651.out 
Tue May 14 17:31:33 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  | 00000000:3F:00.0 Off |                  N/A |
|  0%   33C    P8               1W / 250W |      1MiB / 11264MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
(venv) [kiten@eu-login-14 optimistic_dynamics]$ python launcher.py 
2024-05-14 17:34:19.214743: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
/cluster/home/kiten/copax/venv/lib64/python3.11/site-packages/jax/_src/api_util.py:172: SyntaxWarning: Jitted function has static_argnums=(3, 4), but only accepts 4 positional arguments. This warning will be replaced by an error after 2022-08-20 at the earliest.
  warnings.warn(f"Jitted function has {argnums_name}={argnums}, "
2024-05-14 17:34:22.626090: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:268] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
about to launch 5 jobs with 1 cores each. proceed? [yes/no]no
```

Questions for Bhavy and Lenart:
- 
- In general, it works on CPU: 
    - https://wandb.ai/kiten-ethz/opax_optimistic_April_test2/workspace?nw=nwuserklemensiten
    - Show error when running launcher.py
- Confirm that you are on Python 3.11?
- Confirm TF and numpy version. They are incompatible in Lenarts environment? TF <2.14 is incompatible with python 3.11?
- Currently, I am on TF version 2.16. Should still work, no?
- Error above.
    - Can't find TensorRT
        - I can't install it on Euler explicitly, can I? It's not available as a module. Should come with CUDA/CUDNN or Tensorflow?
        - is it the right tensorflow installation?
    - Random jit error
    - Does not access GPUs
    - But if I submit a batch job with nvidia-smi, I get no errors, so I have access to gpus?
    - Followed the TF instructions and submitted test job on copax/slurm: https://www.tensorflow.org/install/pip#step-by-step_instructions
    - So, I can access GPUs, the main problem seems to be TensorRT. Any tips?
