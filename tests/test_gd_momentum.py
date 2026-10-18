"""Module 1: spectral claims, marginal modes and heavy-ball initialization."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import QuadraticProblem, RunConfig, rotation_2d, run
from coordinate_geometry.gd_momentum import (
    gd_spectral_radius, heavy_ball_is_stable, heavy_ball_spectral_radius,
    optimal_gd_parameters, optimal_heavy_ball_parameters,
)


class SpectralTheoryTests(unittest.TestCase):
    def test_hand_calculated_two_dimensional_example(self):
        d = np.array([0.01, 1.0])
        q = rotation_2d(np.pi / 4)
        problem = QuadraticProblem(d, q)
        initial = q @ np.ones(2)
        for algorithm in ("gd", "heavy_ball"):
            first = run(problem, initial, RunConfig(algorithm, 1.0, 1, beta=0.5))
            assert_allclose(first.final_position, q @ [0.99, 0], atol=1e-14)
            assert_allclose(first.final_loss, 0.0049005, atol=1e-14)
        second = run(problem, initial, RunConfig("heavy_ball", 1.0, 2, beta=0.5))
        assert_allclose(second.final_position, q @ [0.9751, -0.5], atol=1e-14)
        assert_allclose(second.final_loss, 0.12975410005, atol=1e-14)

    def test_gd_objective_bound_and_worst_mode_equality(self):
        d = np.array([0.01, 0.13, 1.0])
        eta, rho = optimal_gd_parameters(d)
        problem = QuadraticProblem(d, np.eye(3))
        for initial in (np.ones(3), np.array([1.0, 0, 0])):
            result = run(problem, initial, RunConfig("gd", eta, 40))
            actual = np.array([entry.relative_loss for entry in result.trace])
            bound = rho ** (2 * np.arange(41))
            self.assertTrue(np.all(actual <= bound + 1e-13))
            if initial[1] == 0:
                assert_allclose(actual, bound, atol=1e-13)
        assert_allclose(gd_spectral_radius(d, eta), rho)

    def test_heavy_ball_schur_condition_matches_companion_eigenvalues(self):
        d = np.array([0.01, 0.4, 1.0])
        for beta in (0.0, 0.2, 0.9):
            boundary = 2 * (1 + beta)
            for eta in (0.1, 0.99 * boundary, 1.01 * boundary):
                with self.subTest(beta=beta, eta=eta):
                    radius = max(max(abs(np.linalg.eigvals(
                        [[1 + beta - eta * lam, -beta], [1, 0]]))) for lam in d)
                    assert_allclose(heavy_ball_spectral_radius(d, eta, beta), radius, atol=1e-13)
                    self.assertEqual(heavy_ball_is_stable(d, eta, beta), radius < 1)
            self.assertFalse(heavy_ball_is_stable(d, boundary, beta))
            assert_allclose(heavy_ball_spectral_radius(d, boundary, beta), 1, atol=1e-13)

    def test_exact_boundary_modes_do_not_converge(self):
        problem = QuadraticProblem(np.ones(1), np.eye(1))
        gd = run(problem, np.ones(1), RunConfig("gd", 2.0, 40))
        assert_allclose([entry.relative_loss for entry in gd.trace], 1.0)
        beta = 0.5
        hb = run(problem, np.ones(1), RunConfig("heavy_ball", 3.0, 80, beta=beta))
        # At eta L=2(1+beta), roots are -1 and -beta. Zero velocity
        # excites the non-decaying mode with amplitude (1+beta)/(1-beta).
        assert_allclose(abs(hb.final_position[0]), 3.0, atol=1e-13)
        self.assertFalse(hb.threshold_retained)

    def test_optimal_heavy_ball_repeated_root_transient(self):
        d = np.array([0.01, 1.0])
        eta, beta, rho = optimal_heavy_ball_parameters(d)
        assert_allclose([eta, beta, rho], [400 / 121, 81 / 121, 9 / 11])
        result = run(QuadraticProblem(d, np.eye(2)), np.array([0.0, 1.0]),
                     RunConfig("heavy_ball", eta, 50, beta=beta))
        t = np.arange(51)
        exact_coordinate = (1 + (1 + rho) * t) * (-rho)**t
        assert_allclose([entry.relative_loss for entry in result.trace], exact_coordinate**2,
                        atol=1e-12, rtol=1e-11)
        self.assertGreater(result.trace[1].relative_loss, 1)

    def test_isotropic_spectrum_converges_in_one_step(self):
        d = np.full(3, 2.0)
        assert_allclose(optimal_gd_parameters(d), [0.5, 0])
        assert_allclose(optimal_heavy_ball_parameters(d), [0.5, 0, 0])
        for algorithm in ("gd", "heavy_ball"):
            result = run(QuadraticProblem(d, np.eye(3)), np.ones(3),
                         RunConfig(algorithm, 0.5, 5, beta=0))
            assert_allclose(result.final_position, 0, atol=1e-14)

    def test_invalid_spectral_inputs(self):
        for values in ([], [0, 1], [-1, 2], [np.inf], [[1, 2]]):
            for helper in (optimal_gd_parameters, optimal_heavy_ball_parameters):
                with self.assertRaises(ValueError):
                    helper(np.asarray(values))
        for eta, beta in ((0, 0.9), (np.nan, 0.9), (1, 1), (1, -0.1)):
            with self.assertRaises(ValueError):
                heavy_ball_spectral_radius(np.ones(2), eta, beta)


if __name__ == "__main__":
    unittest.main()
