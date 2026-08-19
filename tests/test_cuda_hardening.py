import unittest

import torch

import fastgraphcompute
from fastgraphcompute import (
    binned_select_knn_pca,
    index_replacer,
    oc_helper_matrices,
)


class TestCudaHardening(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(fastgraphcompute.__version__, "1.2.0")

    def test_oc_calc_m_not_false_cpu_contract(self):
        truth = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        row_splits = torch.tensor([0, 4], dtype=torch.int64)
        matrix, matrix_not, objects_per_split = oc_helper_matrices(
            truth, row_splits, calc_m_not=False
        )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix_not.shape, (0, 0))
        self.assertEqual(matrix_not.dtype, truth.dtype)
        self.assertEqual(matrix_not.device, truth.device)
        self.assertTrue(torch.equal(objects_per_split, torch.tensor([2])))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pca_forward_backward_on_custom_stream(self):
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            torch.manual_seed(1234)
            coords = torch.randn(
                128, 8, dtype=torch.float32, device=device, requires_grad=True
            )
            row_splits = torch.tensor([0, 128], dtype=torch.int64, device=device)
            indices, distances = binned_select_knn_pca(
                8, coords, row_splits, max_bin_dims=5, pca_subsample=128
            )
            custom_grad = torch.autograd.grad(
                distances.sum(), coords, retain_graph=True
            )[0]

            selected = coords[indices.detach()]
            reference_distances = (
                coords.unsqueeze(1) - selected
            ).square().sum(dim=-1)
            reference_grad = torch.autograd.grad(
                reference_distances.sum(), coords
            )[0]

        torch.cuda.current_stream(device).wait_stream(stream)

        full_distances = (
            coords.detach().unsqueeze(1) - coords.detach().unsqueeze(0)
        ).square().sum(dim=-1)
        expected_distances = torch.topk(
            full_distances, 8, dim=1, largest=False
        ).values

        self.assertTrue(
            torch.allclose(
                torch.sort(distances, dim=1).values,
                expected_distances,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.sort(distances, dim=1).values,
                torch.sort(reference_distances, dim=1).values,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(custom_grad, reference_grad, atol=1e-5, rtol=1e-4)
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_index_replacer_on_custom_stream(self):
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            replacements = torch.arange(4096, dtype=torch.int64, device=device)
            replacements = replacements.flip(0).contiguous()
            indices = torch.arange(4096, dtype=torch.int64, device=device)
            replaced = index_replacer(indices, replacements)

        torch.cuda.current_stream(device).wait_stream(stream)
        self.assertTrue(torch.equal(replaced, replacements))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_oc_boundary_and_no_m_not_on_custom_stream(self):
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            truth = torch.zeros(1024, dtype=torch.int64, device=device)
            row_splits = torch.tensor([0, 1024], dtype=torch.int64, device=device)
            matrix, matrix_not, objects_per_split = oc_helper_matrices(
                truth, row_splits, calc_m_not=False
            )

        torch.cuda.current_stream(device).wait_stream(stream)
        self.assertEqual(matrix.shape, (1, 1024))
        self.assertEqual(matrix_not.shape, (0, 0))
        self.assertTrue(
            torch.equal(
                torch.sort(matrix[0]).values,
                torch.arange(1024, dtype=torch.int64, device=device),
            )
        )
        self.assertTrue(
            torch.equal(
                objects_per_split,
                torch.tensor([1], dtype=torch.int64, device=device),
            )
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mixed_cpu_cuda_inputs_are_rejected(self):
        indices = torch.arange(32, dtype=torch.int64, device="cuda")
        replacements = torch.arange(32, dtype=torch.int64, device="cpu")

        with self.assertRaisesRegex(
            torch.jit.Error, "Tensors must be on the same device"
        ):
            index_replacer(indices, replacements)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Two CUDA devices required",
    )
    def test_cuda_guard_and_cross_device_rejection(self):
        with torch.cuda.device(0):
            indices = torch.arange(32, dtype=torch.int64, device="cuda:1")
            replacements = torch.arange(
                32, dtype=torch.int64, device="cuda:1"
            ).flip(0).contiguous()
            replaced = index_replacer(indices, replacements)
            self.assertEqual(replaced.device.index, 1)
            self.assertTrue(torch.equal(replaced, replacements))

            with self.assertRaisesRegex(
                RuntimeError, "same device as to_be_replaced"
            ):
                index_replacer(
                    indices,
                    torch.arange(32, dtype=torch.int64, device="cuda:0"),
                )


if __name__ == "__main__":
    unittest.main()
