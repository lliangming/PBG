import sys

import mmcv
import torch
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    try:
        env_info = collect_base_env()
    except UnicodeDecodeError:
        # Windows compiler output may use a different console encoding.
        env_info = {
            'sys.platform': sys.platform,
            'Python': sys.version,
            'PyTorch': torch.__version__,
            'CUDA available': torch.cuda.is_available(),
            'CUDA runtime': torch.version.cuda,
            'MMCV': mmcv.__version__,
        }
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
