import unittest

import torch

from model.object_conditioned import (
    center_gaussian_object_weights,
    hard_neighborhood_weights,
    normalized_map_entropy,
    pair_conditioned_attribute_logits,
    shuffled_batch_weights,
    soft_object_weights,
    target_object_attribute_logits,
    uniform_object_weights,
    wrong_object_weights,
)


class ObjectConditionedTest(unittest.TestCase):

    def test_soft_weights_normalize_each_object(self):
        similarities = torch.randn(2, 4, 3)
        weights = soft_object_weights(similarities, temperature=0.2)
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), torch.ones(2, 3), atol=1e-6))

    def test_hard_neighborhood_is_centered_and_normalized(self):
        similarities = torch.zeros(1, 9, 1)
        similarities[:, 4, :] = 3.0
        weights = hard_neighborhood_weights(similarities, kernel=3)
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 9.0)))

    def test_hard_neighborhood_handles_border(self):
        similarities = torch.zeros(1, 9, 1)
        similarities[:, 0, :] = 3.0
        weights = hard_neighborhood_weights(similarities, kernel=3)
        self.assertEqual(int((weights > 0).sum()), 4)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_pair_pooling_uses_the_pairs_object_and_attribute(self):
        attributes = torch.tensor([[[1.0, 10.0], [3.0, 20.0]]])
        objects = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        pairs = torch.tensor([[0, 0], [1, 1]])
        logits = pair_conditioned_attribute_logits(
            attributes, objects, pairs, logit_scale=2.0)
        self.assertTrue(torch.allclose(logits, torch.tensor([[2.0, 40.0]])))

    def test_entropy_detects_uniform_and_peaked_maps(self):
        uniform = torch.full((1, 4, 1), 0.25)
        peaked = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
        self.assertAlmostEqual(float(normalized_map_entropy(uniform)), 1.0, places=6)
        self.assertAlmostEqual(float(normalized_map_entropy(peaked)), 0.0, places=6)

    def test_uniform_weights_ignore_object_scores(self):
        similarities = torch.randn(2, 9, 3)
        weights = uniform_object_weights(similarities)
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 9.0)))

    def test_center_gaussian_is_normalized_and_object_independent(self):
        similarities = torch.randn(2, 16, 3)
        weights = center_gaussian_object_weights(similarities)
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), torch.ones(2, 3), atol=1e-6))
        self.assertTrue(torch.allclose(weights[:, :, 0], weights[:, :, 2]))
        self.assertGreater(float(weights[0, 5, 0]), float(weights[0, 0, 0]))

    def test_wrong_object_weights_are_deranged(self):
        weights = torch.arange(24.0).reshape(1, 4, 6)
        wrong = wrong_object_weights(weights)
        self.assertTrue(torch.equal(wrong[:, :, 0], weights[:, :, 1]))
        self.assertFalse(torch.equal(wrong[:, :, 0], weights[:, :, 0]))

    def test_shuffled_weights_roll_images(self):
        weights = torch.arange(24.0).reshape(2, 4, 3)
        shuffled = shuffled_batch_weights(weights)
        self.assertTrue(torch.equal(shuffled[0], weights[1]))
        self.assertTrue(torch.equal(shuffled[1], weights[0]))

    def test_target_object_attribute_logits_use_target_mask(self):
        attributes = torch.tensor([[[1.0, 10.0], [3.0, 20.0]]])
        objects = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        logits = target_object_attribute_logits(
            attributes, objects, torch.tensor([1]), logit_scale=2.0)
        self.assertTrue(torch.allclose(logits, torch.tensor([[6.0, 40.0]])))


if __name__ == "__main__":
    unittest.main()
