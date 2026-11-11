from .test import multi_gpu_test, single_gpu_test
from .train import set_random_seed, train_segmentor

__all__ = [
    "multi_gpu_test", "single_gpu_test", "set_random_seed", "train_segmentor"
]
