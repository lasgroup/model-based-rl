import pickle
from etils import epath
from typing import Any


def load_params(path: str) -> Any:
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))


def metrics_to_float(my_dict: dict) -> dict:
    for key, value in my_dict.items():
        my_dict[key] = float(value)
    return my_dict
