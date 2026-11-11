"""Catch scratch-memory growth with event count without requiring a GPU."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from revision_campaign.backends import Backend


class FaissResourcesTest(unittest.TestCase):
    def test_segment_indexes_share_one_resource_and_release_indexes_first(self):
        resources, indexes = [], []
        def resource():
            item = object()
            resources.append(item)
            return item
        def gpu_index(owner, device, cpu):
            self.assertIs(owner, resources[0])
            self.assertEqual(device, 0)
            index = SimpleNamespace(add=lambda part: None)
            indexes.append(index)
            return index
        fake = SimpleNamespace(StandardGpuResources=resource,
            IndexFlatL2=lambda d: d, index_cpu_to_gpu=gpu_index)
        backend = Backend.__new__(Backend)
        backend.name = "faiss"
        backend.resources, backend.indexes = [], []
        backend.parts = [SimpleNamespace(shape=(1024, 8), device=SimpleNamespace(index=0))
                         for _ in range(64)]
        with patch.dict(sys.modules, {"faiss": fake}):
            backend.build({})
        self.assertEqual(len(resources), 1)
        self.assertEqual(len(backend.indexes), 64)
        self.assertEqual(len(backend.resources), 1)
        backend.close()
        self.assertEqual(backend.indexes, [])
        self.assertEqual(backend.resources, [])
