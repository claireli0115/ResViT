import os
from glob import glob
from typing import List, Sequence, Tuple

import numpy as np
import torch

from data.base_dataset import BaseDataset

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - nibabel is an optional dependency
    nib = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _strip_nii_extension(path: str) -> str:
    """Return the filename stem without NIfTI-specific extensions."""
    basename = os.path.basename(path)
    if basename.endswith('.nii.gz'):
        basename = basename[:-7]
    elif basename.endswith('.nii'):
        basename = basename[:-4]
    return basename


def _collect_nii_files(directory: str) -> List[str]:
    """Collect all NIfTI files from ``directory`` (both .nii and .nii.gz)."""
    patterns = ('*.nii', '*.nii.gz')
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(os.path.join(directory, pattern)))
    return sorted(set(files))


class NiiAlignedDataset(BaseDataset):
    """Dataset that pairs aligned NIfTI volumes from ``A`` and ``B`` folders."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--slice_count',
            type=int,
            default=0,
            help='Number of slices to retain from each volume (0 keeps all slices).',
        )
        return parser

    def name(self):
        return 'NiiAlignedDataset'

    def initialize(self, opt):
        if nib is None:
            raise ImportError(
                'nibabel is required for NIfTI datasets but could not be imported.'
            ) from _IMPORT_ERROR

        self.opt = opt
        self.root = opt.dataroot
        phase_dir = os.path.join(self.root, opt.phase)
        self.dir_A = os.path.join(phase_dir, 'A')
        self.dir_B = os.path.join(phase_dir, 'B')

        if not os.path.isdir(self.dir_A) or not os.path.isdir(self.dir_B):
            raise FileNotFoundError(
                'Expected paired directories "A" and "B" under %s' % phase_dir
            )

        a_files = _collect_nii_files(self.dir_A)
        b_files = _collect_nii_files(self.dir_B)

        a_map = {_strip_nii_extension(path): path for path in a_files}
        b_map = {_strip_nii_extension(path): path for path in b_files}

        shared_keys = sorted(set(a_map) & set(b_map))
        if not shared_keys:
            raise RuntimeError(
                'No paired NIfTI volumes were found in %s and %s' % (self.dir_A, self.dir_B)
            )
        missing_in_a = sorted(set(b_map) - set(a_map))
        missing_in_b = sorted(set(a_map) - set(b_map))
        if missing_in_a or missing_in_b:
            raise RuntimeError(
                'Mismatched NIfTI files between %s and %s. Missing in A: %s. Missing in B: %s.'
                % (self.dir_A, self.dir_B, missing_in_a, missing_in_b)
            )

        self.paired_paths: Sequence[Tuple[str, str]] = [
            (a_map[key], b_map[key]) for key in shared_keys
        ]

    def __getitem__(self, index):
        a_path, b_path = self.paired_paths[index]
        a_tensor = self._load_volume(a_path)
        b_tensor = self._load_volume(b_path)

        return {'A': a_tensor, 'B': b_tensor, 'A_paths': a_path, 'B_paths': b_path}

    def __len__(self):
        return len(self.paired_paths)

    def _load_volume(self, path: str) -> torch.Tensor:
        volume = nib.load(path).get_fdata(dtype=np.float32)
        if volume.ndim < 3:
            raise ValueError('Expected NIfTI volume to have at least 3 dimensions: %s' % path)

        # Move slice axis to the front to create a channels-first tensor.
        volume = np.moveaxis(volume, -1, 0)

        if self.opt.slice_count:
            slices = volume.shape[0]
            desired = min(self.opt.slice_count, slices)
            start = (slices - desired) // 2
            end = start + desired
            volume = volume[start:end]

        volume = self._normalize_volume(volume)
        tensor = torch.from_numpy(volume)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _normalize_volume(volume: np.ndarray) -> np.ndarray:
        v_min = float(volume.min())
        v_max = float(volume.max())
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
            volume = volume * 2.0 - 1.0
        else:
            volume = np.zeros_like(volume)
        return volume.astype(np.float32, copy=False)
