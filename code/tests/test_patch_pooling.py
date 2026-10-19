import unittest

import torch

from model.patch_pooling import sparse_patch_logits


class SparsePatchPoolingTest(unittest.TestCase):

    def test_patch_max_selects_best_token(self):
        patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        logits = sparse_patch_logits(patches, text, 2.0, mode="patch_max")
        self.assertTrue(torch.allclose(logits, torch.tensor([[2.0, 2.0]])))

    def test_excluding_cls_changes_selection(self):
        patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        text = torch.tensor([[[1.0, 0.0]]])
        with_cls = sparse_patch_logits(
            patches, text, 1.0, mode="patch_max", include_cls=True)
        without_cls = sparse_patch_logits(
            patches, text, 1.0, mode="patch_max", include_cls=False)
        self.assertEqual(with_cls.item(), 1.0)
        self.assertEqual(without_cls.item(), 0.0)

    def test_topk_one_matches_max(self):
        torch.manual_seed(3)
        patches = torch.randn(2, 5, 4)
        text = torch.randn(2, 3, 4)
        max_logits = sparse_patch_logits(
            patches, text, 3.0, mode="patch_max", include_cls=False)
        topk_logits = sparse_patch_logits(
            patches,
            text,
            3.0,
            mode="patch_topk_lse",
            include_cls=False,
            topk=1,
        )
        self.assertTrue(torch.allclose(max_logits, topk_logits, atol=1e-6))

    def test_topk_lse_backpropagates_to_multiple_tokens(self):
        patches = torch.tensor(
            [[[0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]]], requires_grad=True)
        text = torch.tensor([[[1.0, 0.0]]])
        logits = sparse_patch_logits(
            patches,
            text,
            1.0,
            mode="patch_topk_lse",
            include_cls=True,
            topk=2,
            lse_temperature=0.2,
        )
        logits.sum().backward()
        token_grad = patches.grad.abs().sum(dim=-1).squeeze(0)
        self.assertEqual(int((token_grad > 0).sum()), 2)


if __name__ == "__main__":
    unittest.main()
