import os
import glob

import nibabel as nib
import numpy as np
import torch

from data.base_dataset import BaseDataset


class NiiAlignedDataset(BaseDataset):
    def name(self):
        return 'NiiAlignedDataset'

    def initialize(self, opt):
        self.opt = opt
        self.A_slices = []
        self.B_slices = []
        self.A_paths = []
        self.B_paths = []

        phase_a_dir = os.path.join(opt.dataroot, f"{opt.phase}A")
        phase_b_dir = os.path.join(opt.dataroot, f"{opt.phase}B")

        if not os.path.isdir(phase_a_dir):
            raise FileNotFoundError(f"Phase A directory not found: {phase_a_dir}")
        if not os.path.isdir(phase_b_dir):
            raise FileNotFoundError(f"Phase B directory not found: {phase_b_dir}")

        subject_dirs = self._collect_subject_dirs(phase_a_dir)

        for subject_dir in subject_dirs:
            subject_relpath = os.path.relpath(subject_dir, phase_a_dir)
            subject_name = subject_relpath.replace(os.sep, '/')
            target_subject_dir = os.path.join(phase_b_dir, subject_relpath)

            if not os.path.isdir(target_subject_dir):
                raise FileNotFoundError(
                    f"Missing corresponding target directory for subject '{subject_name}': {target_subject_dir}"
                )

            modalities = {
                't1n': self._find_modality(subject_dir, 't1n'),
                't2w': self._find_modality(subject_dir, 't2w'),
                't2f': self._find_modality(subject_dir, 't2f'),
                't1c': self._find_modality(target_subject_dir, 't1c'),
            }

            missing = [key for key, value in modalities.items() if value is None]
            if missing:
                missing_str = ', '.join(missing)
                raise FileNotFoundError(
                    f"Missing modality '{missing_str}' for subject '{subject_name}'"
                )

            t1n = self._load_volume(modalities['t1n'])
            t2w = self._load_volume(modalities['t2w'])
            t2f = self._load_volume(modalities['t2f'])
            t1c = self._load_volume(modalities['t1c'])

            if t1n.shape != t2w.shape or t1n.shape != t2f.shape:
                raise ValueError(f"Input modalities shape mismatch for subject '{subject_name}'")
            if t1c.shape != t1n.shape:
                raise ValueError(f"Target modality shape mismatch for subject '{subject_name}'")

            A_volume = np.stack([t1n, t2w, t2f], axis=0)
            B_volume = np.expand_dims(t1c, axis=0)

            A_volume, B_volume = self._center_crop(A_volume, B_volume)
            A_volume, B_volume, depth_start = self._select_depth_slices(A_volume, B_volume)

            for depth_idx in range(A_volume.shape[-1]):
                slice_A = np.ascontiguousarray(A_volume[:, :, :, depth_idx])
                slice_B = np.ascontiguousarray(B_volume[:, :, :, depth_idx])

                self.A_slices.append(slice_A)
                self.B_slices.append(slice_B)
                slice_name = f"{subject_name}|slice_{depth_start + depth_idx}"
                self.A_paths.append(slice_name)
                self.B_paths.append(slice_name)

        if not self.A_slices:
            raise RuntimeError("No slices were loaded for NiiAlignedDataset")

        A_stack = np.stack(self.A_slices, axis=0)
        B_stack = np.stack(self.B_slices, axis=0)

        self.A_mean = float(A_stack.mean())
        A_std = float(A_stack.std())
        self.A_std = A_std if A_std > 0 else 1.0
        self.B_mean = float(B_stack.mean())
        B_std = float(B_stack.std())
        self.B_std = B_std if B_std > 0 else 1.0

    def __getitem__(self, index):
        A = torch.from_numpy(self.A_slices[index]).float()
        B = torch.from_numpy(self.B_slices[index]).float()

        A = (A - self.A_mean) / self.A_std
        B = (B - self.B_mean) / self.B_std

        return {
            'A': A,
            'B': B,
            'A_paths': self.A_paths[index],
            'B_paths': self.B_paths[index],
        }

    def __len__(self):
        return len(self.A_slices)

    @staticmethod
    def _collect_subject_dirs(root_dir):
        subject_dirs = []
        for current_root, dirs, files in os.walk(root_dir):
            modality_candidates = []
            for modality in ('t1n', 't2w', 't2f'):
                modality_candidates.extend(
                    glob.glob(os.path.join(current_root, f"{modality}*.nii*"))
                )
            if modality_candidates:
                subject_dirs.append(current_root)
        subject_dirs.sort()
        return subject_dirs

    @staticmethod
    def _find_modality(directory, modality):
        patterns = [
            os.path.join(directory, f"{modality}*.nii"),
            os.path.join(directory, f"{modality}*.nii.gz"),
            os.path.join(directory, f"{modality.upper()}*.nii"),
            os.path.join(directory, f"{modality.upper()}*.nii.gz"),
        ]

        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        matches.sort()
        if matches:
            return matches[0]
        return None

    @staticmethod
    def _load_volume(path):
        volume = nib.load(path).get_fdata()
        volume = np.nan_to_num(volume).astype(np.float32)
        return volume

    @staticmethod
    def _center_crop(A_volume, B_volume):
        _, H, W, _ = A_volume.shape
        top = max((H - 192) // 2, 0)
        left = max((W - 192) // 2, 0)
        bottom = min(top + 192, H)
        right = min(left + 192, W)

        if bottom - top < 192:
            top = max(bottom - 192, 0)
            bottom = top + 192
        if right - left < 192:
            left = max(right - 192, 0)
            right = left + 192

        return (
            A_volume[:, top:bottom, left:right, :],
            B_volume[:, top:bottom, left:right, :],
        )

    @staticmethod
    def _select_depth_slices(A_volume, B_volume):
        D = A_volume.shape[-1]
        start = max(D // 2 - 64, 0)
        end = start + 128
        if end > D:
            end = D
            start = max(end - 128, 0)

        return (
            A_volume[:, :, :, start:end],
            B_volume[:, :, :, start:end],
            start,
        )
