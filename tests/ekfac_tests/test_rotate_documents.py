"""Rotating every document's gradient at once gives each one's rotation."""

import torch

from bergson.hessians.eigenvectors import rotate_documents


def test_rotate_documents_matches_each_document():
    generator = torch.Generator().manual_seed(0)
    for n, o, i in [(1, 3, 5), (4, 7, 2), (3, 6, 6)]:
        grads = torch.randn(n, o, i, generator=generator, dtype=torch.float64)
        eigen_g = torch.linalg.qr(
            torch.randn(o, o, generator=generator, dtype=torch.float64)
        )[0]
        eigen_a = torch.linalg.qr(
            torch.randn(i, i, generator=generator, dtype=torch.float64)
        )[0]
        expected = torch.stack([eigen_g.mT @ g @ eigen_a for g in grads])
        torch.testing.assert_close(rotate_documents(grads, eigen_g, eigen_a), expected)
