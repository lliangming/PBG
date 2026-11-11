"""Shared-window dataset ordered by the numeric frame ID in each filename."""
import os.path as osp
import re

from .builder import DATASETS
from .original_shared_window_gas_dataset import OriginalSharedWindowGasDatasetVideo


@DATASETS.register_module()
class NumericSharedWindowGasDatasetVideo(OriginalSharedWindowGasDatasetVideo):
    """Keep the original sampler, ordering each video's frames numerically.

    The last number in the filename stem is the frame ID. This supports
    unpadded IDs, parenthesized IDs and filenames containing dates/video IDs.
    Window strides count available images after sorting, including gaps in IDs.
    """

    def __init__(self, num_frames=3, temporal_stride=5, **kwargs):
        super().__init__(num_frames=num_frames,
                         temporal_stride=temporal_stride, **kwargs)
        self.img_all = []
        for video in self.videolists:
            numbered = []
            for filename in self.imgdic[video]:
                match = re.search(r'(\d+)\D*$', osp.splitext(filename)[0])
                if match is None:
                    raise ValueError(f'No frame number in {video}/{filename}')
                numbered.append((int(match.group(1)), filename))
            numbers = [number for number, _ in numbered]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate frame numbers in video {video}')
            frames = [filename for _, filename in sorted(numbered)]
            self.imgdic[video] = frames
            self.img_all.extend([[video, filename] for filename in frames])
