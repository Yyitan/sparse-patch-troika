import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from model.troika import Troika


class BranchScoringTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.cls = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        self.patch = torch.randn(2, 5, 8)
        self.text = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
        self.model = SimpleNamespace(
            visual_pooling='patch_max', patch_branches='composition',
            patch_include_cls=False, patch_topk=1, patch_lse_temperature=0.07,
            patch_residual_alpha=1.0,
            clip=SimpleNamespace(logit_scale=torch.tensor(3.).log()))

    def test_primitives_keep_cls_while_composition_uses_patches(self):
        expected = torch.einsum('bd,bkd->bk', self.cls, self.text * 3.)
        for branch in (1, 2):
            actual = Troika.compute_visual_logits(
                self.model, self.cls, self.patch, self.text, branch)
            self.assertTrue(torch.equal(actual, expected))
        composition = Troika.compute_visual_logits(
            self.model, self.cls, self.patch, self.text, 0)
        self.assertFalse(torch.allclose(composition, expected))
        self.model.patch_branches = 'all'
        old = Troika.compute_visual_logits(self.model, self.cls, self.patch, self.text, 0)
        self.assertTrue(torch.equal(composition, old))

    def test_residual_patch_score_is_convex_logit_blend(self):
        self.model.patch_residual_alpha = 0.25
        cls_logits = torch.einsum('bd,bkd->bk', self.cls, self.text * 3.)
        self.model.patch_residual_alpha = 1.0
        patch_logits = Troika.compute_visual_logits(
            self.model, self.cls, self.patch, self.text, 0)
        self.model.patch_residual_alpha = 0.25
        actual = Troika.compute_visual_logits(
            self.model, self.cls, self.patch, self.text, 0)
        expected = 0.75 * cls_logits + 0.25 * patch_logits
        self.assertTrue(torch.allclose(actual, expected))

    def test_all_branches_match_previous_version(self):
        source = Path(__file__).resolve().parents[2] / 'code/model/troika.py'
        spec = importlib.util.spec_from_file_location('reference_troika', str(source))
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        self.model.patch_branches = 'all'
        for mode in ('cls', 'patch_max', 'patch_topk_lse'):
            self.model.visual_pooling = mode
            self.model.patch_topk = 4
            expected = reference.Troika.compute_visual_logits(
                self.model, self.cls, self.patch, self.text)
            for branch in (0, 1, 2):
                actual = Troika.compute_visual_logits(
                    self.model, self.cls, self.patch, self.text, branch)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
