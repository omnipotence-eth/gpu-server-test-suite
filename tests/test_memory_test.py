"""Tests for VRAM integrity and allocation diagnostic.

Tests patch the module-level torch reference to run without a real GPU.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.diagnostics.memory_test import (
    _check_vram_allocation,
    _check_vram_pattern_test,
    run_memory_test,
)
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO


class TestVRAMAllocation:
    """Test VRAM allocation and verification logic."""

    def test_allocation_pass_with_cuda(self):
        """Test successful VRAM allocation with mocked CUDA."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = MemoryError
        mock_torch.float32 = "float32"

        # Calculate expected num_elements: (16384 * 90/100) MiB -> bytes -> /4
        target_mib = int(MOCK_GPU_INFO.vram_total_mib * 90 / 100)
        num_elements = (target_mib * 1024 * 1024) // 4
        expected_sum = 42.0 * num_elements

        # Mock tensor
        mock_tensor = MagicMock()
        mock_tensor.nelement.return_value = num_elements
        mock_tensor.element_size.return_value = 4
        mock_tensor.sum.return_value.item.return_value = expected_sum
        mock_torch.ones.return_value = mock_tensor
        mock_torch.cuda.empty_cache.return_value = None

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            result = _check_vram_allocation(MOCK_GPU_INFO, allocation_pct=90)
        assert result.status == TestStatus.PASS

    def test_allocation_skip_no_torch(self):
        """Test graceful skip when torch is None."""
        with patch("src.diagnostics.memory_test.torch", None):
            result = _check_vram_allocation(MOCK_GPU_INFO, allocation_pct=90)
        assert result.status == TestStatus.SKIP
        assert "PyTorch" in result.message

    def test_allocation_skip_no_cuda(self):
        """Test SKIP when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            result = _check_vram_allocation(MOCK_GPU_INFO, allocation_pct=90)
        assert result.status == TestStatus.SKIP
        assert "CUDA not available" in result.message

    def test_allocation_fail_oom(self):
        """Test FAIL when VRAM allocation runs out of memory."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = MemoryError
        mock_torch.float32 = "float32"
        mock_torch.ones.side_effect = MemoryError("CUDA out of memory")

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            result = _check_vram_allocation(MOCK_GPU_INFO, allocation_pct=90)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-300"


class TestVRAMPattern:
    """Test VRAM pattern write/verify logic."""

    def test_pattern_skip_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            result = _check_vram_pattern_test(MOCK_GPU_INFO)
        assert result.status == TestStatus.SKIP

    def test_pattern_skip_no_torch(self):
        with patch("src.diagnostics.memory_test.torch", None):
            result = _check_vram_pattern_test(MOCK_GPU_INFO)
        assert result.status == TestStatus.SKIP

    def test_pattern_pass(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = MemoryError
        mock_torch.float32 = "float32"

        # Mock tensor that returns correct mean for each pattern
        mock_tensor = MagicMock()
        mock_tensor.mean.return_value.item.side_effect = [0.0, 1.0, 3.14159265, -1.0]
        mock_torch.full.return_value = mock_tensor
        mock_torch.cuda.empty_cache.return_value = None

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            result = _check_vram_pattern_test(MOCK_GPU_INFO)
        assert result.status == TestStatus.PASS
        assert "4" in result.message  # All 4 patterns


class TestMemoryTestSuite:
    """Integration test for memory test module."""

    def test_run_produces_results_per_gpu(self, mock_gpu_info, mock_profile):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            results = run_memory_test([mock_gpu_info], mock_profile)
        assert len(results) == 2
        assert all(r.status == TestStatus.SKIP for r in results)

    def test_result_names(self, mock_gpu_info, mock_profile):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("src.diagnostics.memory_test.torch", mock_torch):
            results = run_memory_test([mock_gpu_info], mock_profile)
        names = [r.test_name for r in results]
        assert "memory_test.vram_allocation" in names
        assert "memory_test.pattern_test" in names

    def test_run_no_torch(self, mock_gpu_info, mock_profile):
        """Verify graceful degradation when torch is None."""
        with patch("src.diagnostics.memory_test.torch", None):
            results = run_memory_test([mock_gpu_info], mock_profile)
        assert len(results) == 2
        assert all(r.status == TestStatus.SKIP for r in results)
