from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.matrix_observation_adapter import (  # noqa: E402
    CONTACT_THRESHOLD_N,
    MatrixObservationSourceDTypeError,
    MatrixObservationSourceMissingError,
    MatrixObservationSourceNonFiniteError,
    MatrixObservationSourceShapeError,
    MatrixObservationSourceTypeError,
    adapt_terrain_pair_force_matrix_w,
)


def _source(num_envs: int = 2) -> torch.Tensor:
    return torch.zeros((num_envs, 19, 1, 3), dtype=torch.float32)


def _storage_pointer(value: torch.Tensor) -> int:
    return value.untyped_storage().data_ptr()


def test_adapter_reduces_filter_before_norm_and_preserves_signed_world_xyz() -> None:
    source = _source()
    source[0, 0, 0] = torch.tensor((3.0, -4.0, 12.0))
    source[1, 18, 0] = torch.tensor((-5.0, 0.0, 0.0))

    result = adapt_terrain_pair_force_matrix_w(source)

    assert torch.equal(result.world_xyz[0, 0], torch.tensor((3.0, -4.0, 12.0)))
    assert result.magnitude[0, 0].item() == pytest.approx(13.0)
    assert torch.equal(result.world_xyz[1, 18], torch.tensor((-5.0, 0.0, 0.0)))
    assert result.magnitude[1, 18].item() == pytest.approx(5.0)
    assert result.contact_mask[0, 0]
    assert result.contact_mask[1, 18]


def test_zero_and_exact_threshold_are_not_contacts_but_values_are_unchanged() -> None:
    source = _source(1)
    threshold = torch.tensor(CONTACT_THRESHOLD_N, dtype=torch.float32)
    source[0, 1, 0, 0] = threshold
    source[0, 2, 0, 0] = torch.nextafter(threshold, torch.tensor(torch.inf))

    result = adapt_terrain_pair_force_matrix_w(source)

    assert result.magnitude[0, 0].item() == 0.0
    assert not result.contact_mask[0, 0]
    assert torch.equal(result.world_xyz[0, 1], source[0, 1, 0])
    assert torch.equal(result.magnitude[0, 1], threshold)
    assert not result.contact_mask[0, 1]
    assert result.contact_mask[0, 2]


@pytest.mark.parametrize("invalid", [None, [[[[0.0, 0.0, 0.0]]]]])
def test_missing_and_non_tensor_sources_fail_closed(invalid: object | None) -> None:
    expected = (
        MatrixObservationSourceMissingError
        if invalid is None
        else MatrixObservationSourceTypeError
    )
    with pytest.raises(expected):
        adapt_terrain_pair_force_matrix_w(invalid)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 19, 3),
        (2, 18, 1, 3),
        (2, 19, 2, 3),
        (2, 19, 1, 2),
    ],
)
def test_wrong_shapes_fail_closed(shape: tuple[int, ...]) -> None:
    with pytest.raises(MatrixObservationSourceShapeError, match=r"\[N, 19, 1, 3\]"):
        adapt_terrain_pair_force_matrix_w(torch.zeros(shape, dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_non_float32_sources_fail_closed(dtype: torch.dtype) -> None:
    with pytest.raises(MatrixObservationSourceDTypeError, match="torch.float32"):
        adapt_terrain_pair_force_matrix_w(torch.zeros((2, 19, 1, 3), dtype=dtype))


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_sources_fail_closed(invalid: float) -> None:
    source = _source()
    source[0, 0, 0, 0] = invalid
    with pytest.raises(MatrixObservationSourceNonFiniteError, match="finite"):
        adapt_terrain_pair_force_matrix_w(source)


def test_noncontiguous_source_is_supported_without_copying_or_mutating_source() -> None:
    base = torch.arange(2 * 19 * 6, dtype=torch.float32).reshape(2, 19, 1, 6)
    source = base[..., ::2]
    assert source.shape == (2, 19, 1, 3)
    assert not source.is_contiguous()

    before_shape = source.shape
    before_stride = source.stride()
    before_storage = _storage_pointer(source)
    before_version = source._version
    before_values = source.clone()

    result = adapt_terrain_pair_force_matrix_w(source)

    assert source.shape == before_shape
    assert source.stride() == before_stride
    assert _storage_pointer(source) == before_storage
    assert source._version == before_version
    assert torch.equal(source, before_values)
    assert torch.equal(result.world_xyz, source[:, :, 0, :])


def test_output_contract_preserves_dtype_device_shapes_and_does_not_alias_source() -> None:
    source = torch.randn((3, 19, 1, 3), dtype=torch.float32)
    result = adapt_terrain_pair_force_matrix_w(source)

    assert result.world_xyz.shape == (3, 19, 3)
    assert result.magnitude.shape == (3, 19)
    assert result.contact_mask.shape == (3, 19)
    assert result.world_xyz.dtype is source.dtype
    assert result.magnitude.dtype is source.dtype
    assert result.contact_mask.dtype is torch.bool
    assert result.world_xyz.device == source.device
    assert result.magnitude.device == source.device
    assert result.contact_mask.device == source.device
    assert _storage_pointer(result.world_xyz) != _storage_pointer(source)
    assert _storage_pointer(result.magnitude) != _storage_pointer(source)
    assert _storage_pointer(result.contact_mask) != _storage_pointer(source)
