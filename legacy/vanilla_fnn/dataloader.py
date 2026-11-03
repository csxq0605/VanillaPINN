"""Utilities for wrapping OpenFOAM outputs in PyTorch Dataset/DataLoader abstractions."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import read_points_data, read_result_data


BatchDict = Dict[str, Union[str, torch.Tensor]]
Batch = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, BatchDict]]


class PINNDataset(Dataset):
    """Generic dataset that exposes spatial coordinates with optional supervision.

    Parameters
    ----------
    coordinates:
        Array-like of shape ``(N, D)`` storing spatial points.
    targets:
        Optional array-like of shape ``(N, M)`` storing supervised labels (e.g. flow
        variables ``p``, ``U``, ``T``). When omitted the dataset emits coordinates only.
    condition_types:
        Optional array-like of shape ``(N,)`` describing boundary-condition tags such as
        ``"internal"`` or ``"inlet"``. When supplied the dataset returns a dictionary in
        the second field of each sample instead of raw targets so that downstream loss
        functions can branch on condition type.
    dtype:
        Torch dtype used to cast numeric arrays.
    device:
        Optional device hint. If provided the tensors are moved eagerly; otherwise they
        stay on CPU and can be moved later inside the training loop.
    """

    def __init__(
        self,
        coordinates: Union[np.ndarray, Sequence[Sequence[float]], torch.Tensor],
        targets: Optional[Union[np.ndarray, Sequence[Sequence[float]], torch.Tensor]] = None,
        condition_types: Optional[Union[Sequence[str], np.ndarray]] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        coords_np = _ensure_2d_array(coordinates)
        self.inputs = torch.as_tensor(coords_np, dtype=dtype)

        if targets is not None:
            targets_np = _ensure_2d_array(targets)
            if targets_np.shape[0] != coords_np.shape[0]:
                raise ValueError(
                    "targets and coordinates must contain the same number of samples "
                    f"(got {targets_np.shape[0]} vs {coords_np.shape[0]})"
                )
            self.targets = torch.as_tensor(targets_np, dtype=dtype)
        else:
            self.targets = None

        if condition_types is not None:
            condition_types = np.asarray(condition_types)
            if condition_types.shape[0] != coords_np.shape[0]:
                raise ValueError(
                    "condition_types must align with coordinates length "
                    f"(got {condition_types.shape[0]} vs {coords_np.shape[0]})"
                )
            # Store as string list to preserve readability; default collate keeps dtype=object.
            self.condition_types: Optional[List[str]] = condition_types.tolist()
        else:
            self.condition_types = None

        if device is not None:
            self.inputs = self.inputs.to(device)
            if self.targets is not None:
                self.targets = self.targets.to(device)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Batch]:
        inputs = self.inputs[idx]

        if self.condition_types is None:
            if self.targets is None:
                return inputs
            return inputs, self.targets[idx]
        
        return inputs, self.condition_types[idx]


def load_unsupervised_dataset(
    points_path: str,
    points_filename: str,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> PINNDataset:
    """Create a :class:`PINNDataset` by loading point coordinates only.
    Parameters
    ----------
    points_path:
        Path to the OpenFOAM ``points`` file.                       
    points_filename:
        Name of the OpenFOAM ``points`` file.
    dtype:
        Torch dtype used to cast numeric arrays.
    device:
        Optional device hint. If provided the tensors are moved eagerly; otherwise they
        stay on CPU and can be moved later inside the training loop.
    Returns
    -------
    PINNDataset:
        Dataset wrapping the coordinates from the OpenFOAM ``points`` file.
    """
    points_dict = read_points_data(points_path, points_filename)
    condition_types = []
    for key, arr in points_dict.items():
        condition_types.extend([key] * arr.shape[0])
    coordinates = np.vstack(list(points_dict.values()))

    coordinates = torch.tensor(coordinates, dtype=dtype)

    return coordinates, condition_types
    # return PINNDataset(coordinates, condition_types=condition_types, dtype=dtype, device=device)


def load_supervised_dataset(
    result_path: str,
    result_time: str,
    points_path: str,
    points_filename: str,
    region: str = "internal",
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> PINNDataset:
    """Create a :class:`PINNDataset` by pairing point coordinates with field results.

    The helper assumes that the OpenFOAM field file (``result_filename``) and the
    corresponding coordinate file (``points_filename``) describe the same ordering of
    samples inside ``region`` (for example ``internal`` cells)."""

    points_dict = read_points_data(points_path, points_filename)
    if region not in points_dict:
        raise KeyError(f"Region '{region}' not found in point file '{points_filename}'.")

    coordinates = points_dict[region]
    t = np.full((coordinates.shape[0], 1), fill_value=float(result_time))
    coordinates = np.hstack((coordinates, t))
    results = read_result_data(result_path, result_time, "U")
    for i, name in enumerate(["rho", "p", "T"]):
        results = np.hstack((results, read_result_data(result_path, result_time, name)))

    if results.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "Mismatch between number of coordinates and result entries: "
            f"{coordinates.shape[0]} vs {results.shape[0]}."
        )

    return PINNDataset(coordinates, targets=results, dtype=dtype, device=device)


def load_predict_dataset(
    points_path: str,
    points_filename: str,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> PINNDataset:
    """Create a :class:`PINNDataset` by loading point coordinates only.
    Parameters
    ----------
    points_path:
        Path to the OpenFOAM ``points`` file.                       
    points_filename:
        Name of the OpenFOAM ``points`` file.
    dtype:
        Torch dtype used to cast numeric arrays.
    device:
        Optional device hint. If provided the tensors are moved eagerly; otherwise they
        stay on CPU and can be moved later inside the training loop.
    Returns
    -------
    PINNDataset:
        Dataset wrapping the coordinates from the OpenFOAM ``points`` file.
    """
    points_dict = read_points_data(points_path, points_filename)
    coordinates = points_dict["internal"]

    return PINNDataset(coordinates, dtype=dtype, device=device)


def make_loader(dataset: Dataset, shuffle_flag: bool, batch_size: int = 100, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
	return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle_flag,
			num_workers=num_workers,
			pin_memory=(dataset.inputs.device.type == "cuda") if isinstance(dataset, PINNDataset) else False,
			drop_last=drop_last,
	)


def _ensure_2d_array(array_like: Union[np.ndarray, Sequence[Sequence[float]], torch.Tensor]) -> np.ndarray:
	array = np.asarray(array_like, dtype=np.float64)
	if array.ndim == 1:
		array = array[:, None]
	if array.ndim != 2:
		raise ValueError("Expected a 2D array or something that can be viewed as (N, D).")
	return array


__all__ = [
    "PINNDataset",
    "load_unsupervised_dataset",
    "load_supervised_dataset",
    "make_loader",
]